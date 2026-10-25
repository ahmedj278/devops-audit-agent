"""The LLM-directed reasoning loop.

The model receives the inspection tools plus three control tools and chooses every
action itself.  Python only (1) executes the requested tool, (2) feeds the observation
back, (3) enforces budgets and (4) validates findings before they are accepted.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from findings import CATEGORIES, SEVERITIES, FindingStore
from llm_client import LLMClient, LLMError
from prompts import INITIAL_USER_TEMPLATE, REVIEW_GATE_TEMPLATE, SYSTEM_PROMPT
from redaction import redact
from tools import RepoTools, tool_definitions

log = logging.getLogger("devops_agent.loop")

MAX_TOOL_CALLS_PER_TURN = 8
COMPACT_KEEP_RECENT = 8
ELIDED_NOTE = "…[older tool output elided to save context; re-run the tool if you need it again]"


def control_tool_definitions() -> list[dict]:
    return [
        {
            "name": "report_violation",
            "description": "Record ONE confirmed, evidence-backed violation. May be rejected with a reason (then fix or drop). "
                           "Use file=null/line=null for repository-wide or missing-file problems; for 'missing' claims "
                           "list the globs you checked in missing_paths.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "severity": {"type": "string", "enum": list(SEVERITIES)},
                    "file": {"type": ["string", "null"], "description": "Repo-relative path or null"},
                    "line": {"type": ["integer", "null"], "description": "1-based line you have seen, or null"},
                    "description": {"type": "string", "description": "Short title of the problem"},
                    "evidence": {"type": "string", "description": "What you observed (no secret values)"},
                    "recommendation": {"type": "string", "description": "Concrete remediation"},
                    "missing_paths": {"type": "array", "items": {"type": "string"},
                                      "description": "Globs asserted NOT to exist (verified by the system)"},
                },
                "required": ["category", "severity", "file", "line", "description", "evidence", "recommendation"],
            },
        },
        {
            "name": "withdraw_violation",
            "description": "Remove a previously recorded finding (false positive / duplicate).",
            "input_schema": {"type": "object",
                             "properties": {"rule_id": {"type": "string"}, "reason": {"type": "string"}},
                             "required": ["rule_id", "reason"]},
        },
        {
            "name": "finish_audit",
            "description": "Call when the investigation is complete. The first call may trigger a self-review of your findings.",
            "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": []},
        },
    ]


@dataclass
class AuditConfig:
    max_steps: int = 40
    max_wall_seconds: float = 900.0
    max_tokens: int = 4096
    compaction_chars: int = 150_000
    max_nudges: int = 2
    min_inspections: int = 3


@dataclass
class AuditResult:
    violations: list[dict] = field(default_factory=list)
    finished: bool = False
    stop_reason: str = ""
    steps: int = 0
    tool_usage: Counter = field(default_factory=Counter)
    llm_error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class AuditAgent:
    def __init__(self, llm: LLMClient, tools: RepoTools, store: FindingStore,
                 config: AuditConfig | None = None) -> None:
        self.llm, self.tools, self.store = llm, tools, store
        self.cfg = config or AuditConfig()
        self.tool_defs = tool_definitions() + control_tool_definitions()
        self._review_done = False
        self._early_finish_rejected = False
        self._inspections = 0

    # ------------------------------------------------------------------ main loop
    def run(self, repo_url: str) -> AuditResult:
        res = AuditResult()
        messages: list[dict] = [{
            "role": "user",
            "content": INITIAL_USER_TEMPLATE.format(repo_url=repo_url, max_steps=self.cfg.max_steps),
        }]
        deadline = time.monotonic() + self.cfg.max_wall_seconds
        nudges = 0

        for step in range(1, self.cfg.max_steps + 1):
            if time.monotonic() > deadline:
                res.stop_reason = "time_budget_exhausted"
                break
            res.steps = step
            self._compact(messages)
            try:
                resp = self.llm.complete(SYSTEM_PROMPT, messages, self.tool_defs, self.cfg.max_tokens)
            except LLMError as e:
                res.llm_error = redact(str(e))[:500]
                res.stop_reason = "llm_error"
                log.error("LLM failure at step %d: %s", step, res.llm_error)
                break
            res.input_tokens += resp.usage.get("input_tokens", 0)
            res.output_tokens += resp.usage.get("output_tokens", 0)
            messages.append({"role": "assistant", "content": resp.content or [{"type": "text", "text": "(no output)"}]})

            if not resp.tool_calls:
                nudges += 1
                if nudges > self.cfg.max_nudges:
                    res.stop_reason = "model_stopped_without_finish"
                    break
                messages.append({"role": "user", "content":
                                 "You did not call a tool. Continue investigating with the tools, or call finish_audit if done."})
                continue

            results: list[dict] = []
            finished = False
            for idx, call in enumerate(resp.tool_calls):
                if idx >= MAX_TOOL_CALLS_PER_TURN:
                    results.append(self._result(call.id, f"Skipped: at most {MAX_TOOL_CALLS_PER_TURN} tool calls per turn.", True))
                    continue
                res.tool_usage[call.name] += 1
                log.info("step %d: %s(%s)", step, call.name, self._brief_args(call.input))
                content, is_error, done = self._dispatch(call.name, call.input)
                finished = finished or done
                results.append(self._result(call.id, content, is_error))
            user_blocks: list[dict] = list(results)
            remaining = self.cfg.max_steps - step
            if not finished and remaining in (3, 1):
                user_blocks.append({"type": "text", "text":
                                    f"[budget] {remaining} turn(s) left. Record remaining confirmed findings and call finish_audit."})
            messages.append({"role": "user", "content": user_blocks})
            if finished:
                res.finished, res.stop_reason = True, "finished"
                break
        else:
            res.stop_reason = "max_steps_reached"
        if not res.stop_reason:
            res.stop_reason = "stopped"
        res.violations = self.store.finalize()
        return res

    # ------------------------------------------------------------------ dispatch
    def _dispatch(self, name: str, args: Any) -> tuple[str, bool, bool]:
        """Returns (content, is_error, finished)."""
        if not isinstance(args, dict):
            return "Error: arguments must be a JSON object", True, False
        if "__invalid_json__" in args:
            return "Error: your tool arguments were not valid JSON; retry the call.", True, False
        if name == "report_violation":
            ok, msg = self.store.add(args)
            return (msg if ok else f"REJECTED: {msg}"), (not ok), False
        if name == "withdraw_violation":
            ok, msg = self.store.withdraw(args.get("rule_id"), args.get("reason"))
            return msg, (not ok), False
        if name == "finish_audit":
            return self._handle_finish()
        result = self.tools.execute(name, args)
        if not result.is_error:
            self._inspections += 1
        return result.content, result.is_error, False

    def _handle_finish(self) -> tuple[str, bool, bool]:
        if self._inspections < self.cfg.min_inspections and not self._early_finish_rejected:
            self._early_finish_rejected = True
            return ("You have barely inspected the repository. Explore it properly (structure, manifests, CI, "
                    "configuration) before finishing, then call finish_audit again."), False, False
        if len(self.store) > 0 and not self._review_done:
            self._review_done = True
            return REVIEW_GATE_TEMPLATE.format(summary=self.store.summary()), False, False
        return "Audit complete.", False, True

    @staticmethod
    def _result(tool_use_id: str, content: str, is_error: bool) -> dict:
        block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
        if is_error:
            block["is_error"] = True
        return block

    @staticmethod
    def _brief_args(args: Any) -> str:
        try:
            return redact(json.dumps(args, default=str))[:160]
        except Exception:
            return "?"

    # ------------------------------------------------------------------ context management
    def _compact(self, messages: list[dict]) -> None:
        """Elide old tool outputs once the transcript grows beyond the budget."""
        def size(m: dict) -> int:
            c = m["content"]
            if isinstance(c, str):
                return len(c)
            return sum(len(str(b.get("content", b.get("text", "")))) for b in c)

        total = sum(size(m) for m in messages)
        if total <= self.cfg.compaction_chars:
            return
        cutoff = len(messages) - COMPACT_KEEP_RECENT
        for m in messages[:cutoff]:
            if m["role"] != "user" or isinstance(m["content"], str):
                continue
            for b in m["content"]:
                if b.get("type") == "tool_result" and len(str(b.get("content", ""))) > 400:
                    b["content"] = str(b["content"])[:300] + "\n" + ELIDED_NOTE
