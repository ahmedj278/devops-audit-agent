import json
import sys
import types
from types import SimpleNamespace

import pytest

from llm_client import (ConfigError, LLMClient, LLMFatalError, LLMResponse, LLMTransientError,
                        LLMUnavailableError, ResilientLLM, create_llm_client, parse_openai_message,
                        to_openai_messages, to_openai_tools)


class Flaky(LLMClient):
    def __init__(self, errors):
        self.errors, self.calls = list(errors), 0

    def complete(self, system, messages, tools, max_tokens=4096):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return LLMResponse("ok", [], "end_turn", [])


def test_retries_transient_then_succeeds():
    sleeps = []
    inner = Flaky([LLMTransientError("429"), LLMTransientError("503")])
    out = ResilientLLM(inner, sleep=sleeps.append).complete("s", [], [])
    assert out.text == "ok" and inner.calls == 3 and len(sleeps) == 2
    assert sleeps[1] >= sleeps[0] * 0.5  # backoff grows (with jitter)


def test_gives_up_after_max_attempts():
    inner = Flaky([LLMTransientError("x")] * 10)
    with pytest.raises(LLMUnavailableError):
        ResilientLLM(inner, max_attempts=3, sleep=lambda s: None).complete("s", [], [])
    assert inner.calls == 3


def test_fatal_errors_not_retried():
    inner = Flaky([LLMFatalError("401")])
    with pytest.raises(LLMFatalError):
        ResilientLLM(inner, sleep=lambda s: None).complete("s", [], [])
    assert inner.calls == 1


def test_config_errors(monkeypatch):
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        create_llm_client(env={})
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        create_llm_client("openai", env={})
    with pytest.raises(ConfigError, match="unsupported"):
        create_llm_client("skynet", env={"ANTHROPIC_API_KEY": "x"})


def test_factory_builds_real_sdk_client_without_network():
    pytest.importorskip("anthropic")  # skipped where the SDK cannot be installed
    c = create_llm_client(env={"ANTHROPIC_API_KEY": "sk-test-not-real-key-12345", "LLM_MODEL": "m"})
    assert isinstance(c, ResilientLLM) and c.inner.model == "m"


def test_missing_sdk_is_a_clear_config_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)  # makes `import anthropic` raise ImportError
    with pytest.raises(ConfigError, match="SDK is not installed"):
        create_llm_client(env={"ANTHROPIC_API_KEY": "sk-test-not-real-key-12345"})


# --- Adapter contract tests against a *fake* SDK module -------------------------------------
# These exercise our request/response/exception-mapping code. They do NOT prove compatibility
# with the real vendor SDKs; run a live audit with a real key for that.
def _fake_anthropic(behavior):
    mod = types.ModuleType("anthropic")

    class APIError(Exception): ...
    class APIConnectionError(APIError): ...
    class APITimeoutError(APIConnectionError): ...
    class APIStatusError(APIError):
        def __init__(self, status_code, message="boom"):
            super().__init__(message)
            self.status_code, self.message = status_code, message
    class RateLimitError(APIStatusError):
        def __init__(self):
            super().__init__(429, "slow down")

    class _Messages:
        def create(self, **kw):
            return behavior(kw)

    class Anthropic:
        def __init__(self, **kw):
            self.kw, self.messages = kw, _Messages()

    for obj in (APIError, APIConnectionError, APITimeoutError, APIStatusError, RateLimitError, Anthropic):
        setattr(mod, obj.__name__, obj)
    return mod


def test_anthropic_adapter_parses_response_and_passes_arguments(monkeypatch):
    seen = {}

    def behavior(kw):
        seen.update(kw)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="checking"),
                     SimpleNamespace(type="tool_use", id="t1", name="list_files", input={"path": "src/"})],
            stop_reason="tool_use", usage=SimpleNamespace(input_tokens=11, output_tokens=7))

    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic(behavior))
    client = create_llm_client(env={"ANTHROPIC_API_KEY": "sk-test-not-real-key-12345", "LLM_MODEL": "model-x"}).inner
    out = client.complete("SYS", [{"role": "user", "content": "hi"}], [{"name": "t"}], 123)
    assert seen["model"] == "model-x" and seen["system"] == "SYS" and seen["max_tokens"] == 123
    assert client.client.kw["max_retries"] == 0  # retries are ours, with backoff
    assert out.tool_calls[0].name == "list_files" and out.tool_calls[0].input == {"path": "src/"}
    assert out.content[1]["type"] == "tool_use" and out.usage == {"input_tokens": 11, "output_tokens": 7}


@pytest.mark.parametrize("make_exc,expected", [
    (lambda m: m.RateLimitError(), LLMTransientError),
    (lambda m: m.APITimeoutError("t"), LLMTransientError),
    (lambda m: m.APIConnectionError("c"), LLMTransientError),
    (lambda m: m.APIStatusError(503), LLMTransientError),
    (lambda m: m.APIStatusError(529), LLMTransientError),
    (lambda m: m.APIStatusError(401, "bad key"), LLMFatalError),
    (lambda m: m.APIStatusError(400, "bad model"), LLMFatalError),
])
def test_anthropic_adapter_error_mapping(monkeypatch, make_exc, expected):
    mod = _fake_anthropic(lambda kw: (_ for _ in ()).throw(make_exc(mod)))
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    client = create_llm_client(env={"ANTHROPIC_API_KEY": "sk-test-not-real-key-12345"}).inner
    with pytest.raises(expected):
        client.complete("s", [], [])


def test_openai_message_conversion_roundtrip():
    msgs = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": [{"type": "text", "text": "looking"},
                                          {"type": "tool_use", "id": "c1", "name": "read_file", "input": {"path": "a"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "data"},
                                     {"type": "text", "text": "budget"}]},
    ]
    out = to_openai_messages("SYS", msgs)
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[2]["tool_calls"][0]["function"]["arguments"] == json.dumps({"path": "a"})
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "data"}
    assert out[4] == {"role": "user", "content": "budget"}


def test_openai_tools_and_response_parsing():
    tools = to_openai_tools([{"name": "t", "description": "d", "input_schema": {"type": "object"}}])
    assert tools[0]["function"]["parameters"] == {"type": "object"}
    msg = SimpleNamespace(content="hi", tool_calls=[
        SimpleNamespace(id="1", function=SimpleNamespace(name="a", arguments='{"x": 1}')),
        SimpleNamespace(id="2", function=SimpleNamespace(name="b", arguments="{broken")),
    ])
    r = parse_openai_message(msg, "tool_calls")
    assert r.tool_calls[0].input == {"x": 1} and "__invalid_json__" in r.tool_calls[1].input
    assert [b["type"] for b in r.content] == ["text", "tool_use", "tool_use"]


# --- Gemini (OpenAI-compatible endpoint) -------------------------------------------------------
def test_gemini_requires_key_and_accepts_google_key_alias():
    from llm_client import GeminiClient
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        create_llm_client("gemini", env={})
    pytest.importorskip("openai")
    c = create_llm_client("gemini", env={"GOOGLE_API_KEY": "fake-google-key-123456"}).inner
    assert isinstance(c, GeminiClient) and c.model == "gemini-2.5-flash"
    assert "generativelanguage.googleapis.com" in str(c.client.base_url)


def test_simplify_schema_removes_type_unions():
    from llm_client import simplify_schema
    out = simplify_schema({"type": "object", "properties": {"f": {"type": ["string", "null"]}, "l": {"type": ["integer", "null"]}}})
    assert out["properties"]["f"]["type"] == "string" and out["properties"]["l"]["type"] == "integer"


def test_all_real_tool_schemas_are_gemini_safe():
    from agent_loop import control_tool_definitions
    from llm_client import to_openai_tools
    from tools import tool_definitions
    blob = json.dumps(to_openai_tools(tool_definitions() + control_tool_definitions(), simplify=True))
    assert '"type": ["' not in blob


def test_missing_tool_call_id_gets_generated_and_tool_name_added():
    msg = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="", function=SimpleNamespace(name="a", arguments="{}"))])
    r = parse_openai_message(msg, "tool_calls")
    assert r.tool_calls[0].id and r.content[0]["id"] == r.tool_calls[0].id
    msgs = [{"role": "assistant", "content": r.content},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": r.tool_calls[0].id, "content": "x"}]}]
    assert to_openai_messages("S", msgs, include_tool_name=True)[-1]["name"] == "a"
    assert "name" not in to_openai_messages("S", msgs)[-1]
