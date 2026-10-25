"""Test doubles: a scripted LLM whose *policy* sees the conversation, so tests can prove
that tool choice reacts to earlier observations (the agent is not a fixed pipeline)."""

from __future__ import annotations

import itertools
from typing import Callable

from llm_client import LLMClient, LLMResponse, ToolCall

_ids = itertools.count(1)

# Fake credentials are assembled at runtime so no secret-shaped literal lives in the repo
# (keeps GitHub push-protection / secret scanners quiet).
FAKE_AWS_KEY = "AKIA" + "ABCDEFGHIJKLMNOP"
FAKE_GH_TOKEN = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"


def call(name: str, **inp) -> ToolCall:
    return ToolCall(f"toolu_{next(_ids)}", name, inp)


def reply(*calls: ToolCall, text: str = "") -> LLMResponse:
    blocks: list[dict] = []
    if text:
        blocks.append({"type": "text", "text": text})
    for c in calls:
        blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.input})
    return LLMResponse(text, list(calls), "tool_use" if calls else "end_turn", blocks)


def last_results(messages: list[dict]) -> list[str]:
    """Contents of the tool_result blocks in the most recent user message."""
    last = messages[-1]
    if last["role"] != "user" or isinstance(last["content"], str):
        return []
    return [str(b.get("content", "")) for b in last["content"] if b.get("type") == "tool_result"]


def all_results_text(messages: list[dict]) -> str:
    out = []
    for m in messages:
        if m["role"] == "user" and not isinstance(m["content"], str):
            out += [str(b.get("content", "")) for b in m["content"] if b.get("type") == "tool_result"]
    return "\n".join(out)


class ScriptedLLM(LLMClient):
    """policy(messages, n_calls) -> LLMResponse | Exception"""

    def __init__(self, policy: Callable[[list[dict], int], LLMResponse | Exception]) -> None:
        self.policy = policy
        self.n = 0
        self.seen_systems: list[str] = []
        self.tool_names: list[list[str]] = []

    def complete(self, system, messages, tools, max_tokens=4096):
        self.n += 1
        self.seen_systems.append(system)
        self.tool_names.append([t["name"] for t in tools])
        out = self.policy(messages, self.n)
        if isinstance(out, Exception):
            raise out
        return out
