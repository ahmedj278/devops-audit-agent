"""Thin LLM provider layer (plain vendor SDKs, no orchestration frameworks).

Internally everything uses one *canonical* message format (Anthropic style):
  assistant content: [{"type":"text",...}, {"type":"tool_use","id","name","input"}]
  user content:      str | [{"type":"tool_result","tool_use_id","content","is_error"}, {"type":"text",...}]
The OpenAI adapter converts to/from this format.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from redaction import register_secret

log = logging.getLogger("devops_agent.llm")

DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o", "gemini": "gemini-2.5-flash"}
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class ConfigError(Exception):
    """Missing or invalid LLM configuration (e.g. no API key)."""


class LLMError(Exception):
    """Base class for LLM failures."""


class LLMTransientError(LLMError):
    """Retryable: rate limit, timeout, 5xx, connection problem."""


class LLMFatalError(LLMError):
    """Not retryable: bad credentials, invalid model, malformed request."""


class LLMUnavailableError(LLMError):
    """Retries exhausted."""


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    content: list[dict] = field(default_factory=list)  # canonical assistant blocks
    usage: dict = field(default_factory=dict)


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int = 4096) -> LLMResponse:
        ...


# --------------------------------------------------------------------------- Anthropic
class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str, timeout: float = 120.0) -> None:
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=0)
        self.model = model

    def complete(self, system, messages, tools, max_tokens=4096) -> LLMResponse:
        a = self._anthropic
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system, messages=messages, tools=tools,
            )
        except (a.APITimeoutError, a.APIConnectionError, a.RateLimitError) as e:
            raise LLMTransientError(f"{type(e).__name__}: {e}") from e
        except a.APIStatusError as e:
            if e.status_code >= 500 or e.status_code in (408, 409, 429, 529):
                raise LLMTransientError(f"HTTP {e.status_code}: {e}") from e
            raise LLMFatalError(f"HTTP {e.status_code}: {getattr(e, 'message', e)}") from e
        blocks: list[dict] = []
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for b in resp.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
                text_parts.append(b.text)
            elif b.type == "tool_use":
                inp = b.input if isinstance(b.input, dict) else {}
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": inp})
                calls.append(ToolCall(b.id, b.name, inp))
        usage = {"input_tokens": getattr(resp.usage, "input_tokens", 0),
                 "output_tokens": getattr(resp.usage, "output_tokens", 0)}
        return LLMResponse("\n".join(text_parts), calls, resp.stop_reason or "", blocks, usage)


# --------------------------------------------------------------------------- OpenAI
def to_openai_messages(system: str, messages: list[dict], include_tool_name: bool = False) -> list[dict]:
    """Convert canonical messages to OpenAI chat-completions messages."""
    out: list[dict] = [{"role": "system", "content": system}]
    names: dict[str, str] = {}
    for m in messages:
        content = m["content"]
        if m["role"] == "assistant":
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text") if isinstance(content, list) else str(content)
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            calls = [b for b in content if isinstance(content, list) and b.get("type") == "tool_use"]
            for c in calls:
                names[c["id"]] = c["name"]
            if calls:
                msg["tool_calls"] = [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": json.dumps(c.get("input", {}))}}
                    for c in calls
                ]
            out.append(msg)
        else:
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
                continue
            texts = []
            for b in content:
                if b.get("type") == "tool_result":
                    msg_t = {"role": "tool", "tool_call_id": b["tool_use_id"], "content": str(b.get("content", ""))}
                    if include_tool_name and b["tool_use_id"] in names:
                        msg_t["name"] = names[b["tool_use_id"]]
                    out.append(msg_t)
                elif b.get("type") == "text":
                    texts.append(b["text"])
            if texts:
                out.append({"role": "user", "content": "\n".join(texts)})
    return out


def simplify_schema(node):
    """Gemini rejects JSON-schema type unions like ["string","null"]; keep the non-null type."""
    if isinstance(node, list):
        return [simplify_schema(x) for x in node]
    if not isinstance(node, dict):
        return node
    out = {k: simplify_schema(v) for k, v in node.items()}
    t = out.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        out["type"] = non_null[0] if non_null else "string"
    return out


def to_openai_tools(tools: list[dict], simplify: bool = False) -> list[dict]:
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": simplify_schema(t["input_schema"]) if simplify else t["input_schema"]}}
            for t in tools]


def parse_openai_message(message: Any, finish_reason: str | None) -> LLMResponse:
    text = message.content or ""
    blocks: list[dict] = [{"type": "text", "text": text}] if text else []
    calls: list[ToolCall] = []
    for tc in getattr(message, "tool_calls", None) or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError
        except ValueError:
            args = {"__invalid_json__": (tc.function.arguments or "")[:200]}
        call_id = tc.id or f"call_{uuid.uuid4().hex[:12]}"  # some providers return empty/duplicate ids
        calls.append(ToolCall(call_id, tc.function.name, args))
        blocks.append({"type": "tool_use", "id": call_id, "name": tc.function.name, "input": args})
    return LLMResponse(text, calls, finish_reason or "", blocks)


class OpenAIClient(LLMClient):
    gemini_mode = False  # GeminiClient flips this

    def __init__(self, api_key: str, model: str, base_url: str | None = None, timeout: float = 120.0) -> None:
        import openai

        self._openai = openai
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        self.model = model

    def complete(self, system, messages, tools, max_tokens=4096) -> LLMResponse:
        o = self._openai
        try:
            token_arg = {"max_tokens": max_tokens} if self.gemini_mode else {"max_completion_tokens": max_tokens}
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=to_openai_messages(system, messages, include_tool_name=self.gemini_mode),
                tools=to_openai_tools(tools, simplify=self.gemini_mode), **token_arg,
            )
        except (o.APITimeoutError, o.APIConnectionError, o.RateLimitError) as e:
            raise LLMTransientError(f"{type(e).__name__}: {e}") from e
        except o.APIStatusError as e:
            if e.status_code >= 500 or e.status_code in (408, 409, 429):
                raise LLMTransientError(f"HTTP {e.status_code}: {e}") from e
            raise LLMFatalError(f"HTTP {e.status_code}: {getattr(e, 'message', e)}") from e
        choice = resp.choices[0]
        out = parse_openai_message(choice.message, choice.finish_reason)
        if resp.usage:
            out.usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
        return out


class GeminiClient(OpenAIClient):
    """Google Gemini through its OpenAI-compatible endpoint (works with the free API key)."""

    gemini_mode = True

    def __init__(self, api_key: str, model: str, timeout: float = 120.0) -> None:
        super().__init__(api_key, model, GEMINI_BASE_URL, timeout)


# --------------------------------------------------------------------------- resilience
class ResilientLLM(LLMClient):
    """Retries transient failures with exponential backoff + jitter."""

    def __init__(self, inner: LLMClient, max_attempts: int = 5, base_delay: float = 2.0,
                 max_delay: float = 60.0, sleep: Callable[[float], None] = time.sleep) -> None:
        self.inner, self.max_attempts = inner, max_attempts
        self.base_delay, self.max_delay, self._sleep = base_delay, max_delay, sleep

    def complete(self, system, messages, tools, max_tokens=4096) -> LLMResponse:
        last: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.inner.complete(system, messages, tools, max_tokens)
            except LLMTransientError as e:
                last = e
                if attempt == self.max_attempts:
                    break
                delay = min(self.max_delay, self.base_delay * 2 ** (attempt - 1)) * (0.5 + random.random() / 2)
                log.warning("LLM transient error (attempt %d/%d): %s - retrying in %.1fs",
                            attempt, self.max_attempts, type(e).__name__, delay)
                self._sleep(delay)
        raise LLMUnavailableError(f"LLM unavailable after {self.max_attempts} attempts: {last}")


def create_llm_client(provider: str | None = None, model: str | None = None,
                      env: dict[str, str] | None = None) -> LLMClient:
    env = os.environ if env is None else env
    provider = (provider or env.get("LLM_PROVIDER") or "anthropic").strip().lower()
    if provider not in DEFAULT_MODELS:
        raise ConfigError(f"unsupported LLM_PROVIDER '{provider}' (use 'anthropic', 'openai' or 'gemini')")
    key_var = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}[provider]
    key = (env.get(key_var) or "").strip()
    if not key and provider == "gemini":
        key = (env.get("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise ConfigError(f"{key_var} is not set. Copy .env.example to .env and add your key (or export it).")
    register_secret(key)
    model = model or env.get("LLM_MODEL") or DEFAULT_MODELS[provider]
    timeout = float(env.get("LLM_REQUEST_TIMEOUT", "120"))
    try:
        if provider == "anthropic":
            inner: LLMClient = AnthropicClient(key, model, timeout)
        elif provider == "gemini":
            inner = GeminiClient(key, model, timeout)
        else:
            inner = OpenAIClient(key, model, env.get("OPENAI_BASE_URL") or None, timeout)
    except ImportError as e:
        raise ConfigError(f"the '{provider}' SDK is not installed: {e}") from e
    log.info("using provider=%s model=%s", provider, model)
    if provider == "gemini":  # free tier has tight per-minute limits: wait longer, retry more
        return ResilientLLM(inner, max_attempts=8, base_delay=8.0, max_delay=65.0)
    return ResilientLLM(inner)
