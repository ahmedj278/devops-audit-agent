"""Violation validation, de-duplication and the JSON report contract.

The LLM proposes findings via ``report_violation``; this module decides whether they
are *defensible* before they enter the report:

* the cited file must exist; a cited line must have been shown to the agent
* absence claims (``missing_paths``) are checked against the real file tree
* secrets are redacted from every text field
* duplicates are rejected; rule IDs are assigned by code (``SEC-001`` ...)
"""

from __future__ import annotations

import difflib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from redaction import redact, redact_obj
from tools import RepoTools, ToolError

CATEGORIES = {
    "repository_hygiene": "HYG",
    "dependency_management": "DEP",
    "code_quality": "QUA",
    "testing": "TST",
    "ci_cd": "CI",
    "security": "SEC",
    "configuration": "CFG",
    "collaboration": "COL",
    "build_release": "BLD",
}
_ALIASES = {
    "repo_hygiene": "repository_hygiene", "hygiene": "repository_hygiene", "repository": "repository_hygiene",
    "dependencies": "dependency_management", "dependency": "dependency_management",
    "quality": "code_quality", "lint": "code_quality", "linting": "code_quality",
    "tests": "testing", "test": "testing",
    "ci": "ci_cd", "cicd": "ci_cd", "github_actions": "ci_cd", "github_actions_ci": "ci_cd", "ci_cd_pipeline": "ci_cd",
    "secrets": "security",
    "config": "configuration",
    "pull_request": "collaboration",
    "build": "build_release", "release": "build_release", "build_and_release": "build_release",
}
SEVERITIES = ("critical", "high", "medium", "low")
_SEV_RANK = {s: i for i, s in enumerate(SEVERITIES)}
FIELD_ORDER = ("rule_id", "category", "severity", "file", "line", "description", "evidence", "recommendation")
RULE_ID_RE = re.compile(r"^[A-Z]{2,4}-\d{3}$")


class FindingError(Exception):
    """A proposed finding was rejected; the message is shown to the model."""


def _text(value: Any, name: str, min_len: int, max_len: int) -> str:
    if not isinstance(value, str) or len(value.strip()) < min_len:
        raise FindingError(f"'{name}' is required (a string of at least {min_len} characters)")
    return " ".join(redact(value).split())[:max_len]


def _normalize_category(value: Any) -> str:
    if not isinstance(value, str):
        raise FindingError(f"'category' must be one of: {', '.join(CATEGORIES)}")
    key = re.sub(r"[\s/&\-]+", "_", value.strip().lower())
    key = _ALIASES.get(key, key)
    if key not in CATEGORIES:
        raise FindingError(f"unknown category '{value}'. Allowed: {', '.join(CATEGORIES)}")
    return key


class FindingStore:
    def __init__(self, tools: RepoTools, max_findings: int = 60) -> None:
        self.tools = tools
        self.max_findings = max_findings
        self._items: list[dict] = []
        self._counters: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._items)

    # ------------------------------------------------------------------ add
    def add(self, args: dict) -> tuple[bool, str]:
        try:
            return True, self._add(args)
        except FindingError as e:
            return False, str(e)
        except ToolError as e:
            return False, f"invalid file path: {e}"

    def _add(self, a: dict) -> str:
        if not isinstance(a, dict):
            raise FindingError("arguments must be an object")
        if len(self._items) >= self.max_findings:
            raise FindingError(f"finding limit ({self.max_findings}) reached; call finish_audit")
        if not self.tools.listed_files:
            raise FindingError("explore the repository first (list_files or find_files) before reporting")
        category = _normalize_category(a.get("category"))
        severity = str(a.get("severity", "")).strip().lower()
        if severity not in _SEV_RANK:
            raise FindingError(f"'severity' must be one of: {', '.join(SEVERITIES)}")
        description = _text(a.get("description"), "description", 8, 200)
        evidence = _text(a.get("evidence"), "evidence", 15, 600)
        recommendation = _text(a.get("recommendation"), "recommendation", 10, 400)

        file_val = a.get("file")
        if isinstance(file_val, str) and file_val.strip().lower() in {"", "null", "none", "n/a"}:
            file_val = None
        line_val = a.get("line")
        if isinstance(line_val, str) and line_val.strip().isdigit():
            line_val = int(line_val)
        if isinstance(line_val, str) and line_val.strip().lower() in {"", "null", "none", "n/a"}:
            line_val = None
        if line_val is not None and (isinstance(line_val, bool) or not isinstance(line_val, int) or line_val < 1):
            raise FindingError("'line' must be a positive integer or null")

        rel: str | None = None
        if file_val is not None:
            if not isinstance(file_val, str):
                raise FindingError("'file' must be a string path or null")
            rel = self.tools.normalize(file_val)
            if not self.tools.exists(rel):
                raise FindingError(
                    f"file '{rel}' does not exist in the repository. If the problem is something that is MISSING, "
                    "set file to null and list the absent path patterns in missing_paths.")
        if line_val is not None:
            if rel is None:
                raise FindingError("'line' requires 'file'; use null for both when the finding is repository-wide")
            if not self.tools.line_seen(rel, line_val):
                raise FindingError(
                    f"line {line_val} of '{rel}' was never shown to you. Use read_file/search_code to view it "
                    "(so the citation is verified), or set line to null.")

        missing = a.get("missing_paths")
        if missing is not None:
            if not isinstance(missing, list) or not all(isinstance(p, str) for p in missing):
                raise FindingError("'missing_paths' must be a list of path globs")
            for pat in missing:
                present = self.tools.matching_paths(pat)
                if present:
                    raise FindingError(
                        f"you claim '{pat}' is missing but it exists ({', '.join(present[:3])}). "
                        "Withdraw the claim or correct it.")

        dup = self._find_duplicate(category, rel, line_val, description)
        if dup:
            raise FindingError(f"duplicate of {dup}; not added. Report each root cause once.")

        self._counters[category] = self._counters.get(category, 0) + 1
        rule_id = f"{CATEGORIES[category]}-{self._counters[category]:03d}"
        self._items.append({
            "rule_id": rule_id, "category": category, "severity": severity, "file": rel, "line": line_val,
            "description": description, "evidence": evidence, "recommendation": recommendation,
        })
        return f"Recorded {rule_id} ({severity})."

    def _find_duplicate(self, category: str, rel: str | None, line: int | None, desc: str) -> str | None:
        for it in self._items:
            if it["category"] != category or it["file"] != rel:
                continue
            if rel is not None and line is not None and it["line"] == line:
                return it["rule_id"]
            ratio = difflib.SequenceMatcher(None, it["description"].lower(), desc.lower()).ratio()
            if ratio >= (0.8 if rel is None else 0.85):
                return it["rule_id"]
        return None

    # ------------------------------------------------------------------ review support
    def withdraw(self, rule_id: Any, reason: Any) -> tuple[bool, str]:
        for i, it in enumerate(self._items):
            if it["rule_id"] == rule_id:
                del self._items[i]
                return True, f"Withdrew {rule_id}."
        return False, f"no recorded finding with id {rule_id!r}"

    def summary(self) -> str:
        if not self._items:
            return "(no findings recorded)"
        rows = []
        for it in self._items:
            loc = f"{it['file']}:{it['line']}" if it["line"] else (it["file"] or "repo-wide")
            rows.append(f"- {it['rule_id']} [{it['severity']}] {loc} - {it['description']} | evidence: {it['evidence']}")
        return "\n".join(rows)

    def finalize(self) -> list[dict]:
        """Sort by severity, renumber IDs sequentially per category, return report-ready dicts."""
        ordered = sorted(enumerate(self._items), key=lambda p: (_SEV_RANK[p[1]["severity"]], p[0]))
        counters: dict[str, int] = {}
        out = []
        for _, it in ordered:
            counters[it["category"]] = counters.get(it["category"], 0) + 1
            row = dict(it)
            row["rule_id"] = f"{CATEGORIES[it['category']]}-{counters[it['category']]:03d}"
            out.append({k: row[k] for k in FIELD_ORDER})
        return out


# ----------------------------------------------------------------------- report contract
def build_report(repository: str, violations: list[dict]) -> dict:
    return redact_obj({"repository": repository, "violations": violations})


def validate_report(report: Any) -> list[str]:
    """Return a list of schema problems (empty list == valid)."""
    errs: list[str] = []
    if not isinstance(report, dict) or set(report) != {"repository", "violations"}:
        return ["top level must be exactly {repository, violations}"]
    if not isinstance(report["repository"], str) or not report["repository"]:
        errs.append("repository must be a non-empty string")
    if not isinstance(report["violations"], list):
        return errs + ["violations must be a list"]
    for i, v in enumerate(report["violations"]):
        p = f"violations[{i}]"
        if not isinstance(v, dict) or set(v) != set(FIELD_ORDER):
            errs.append(f"{p} must have exactly the fields {list(FIELD_ORDER)}")
            continue
        if not isinstance(v["rule_id"], str) or not RULE_ID_RE.match(v["rule_id"]):
            errs.append(f"{p}.rule_id invalid")
        if v["severity"] not in SEVERITIES:
            errs.append(f"{p}.severity invalid")
        if v["file"] is not None and not isinstance(v["file"], str):
            errs.append(f"{p}.file must be string or null")
        if v["line"] is not None and (isinstance(v["line"], bool) or not isinstance(v["line"], int)):
            errs.append(f"{p}.line must be integer or null")
        for k in ("category", "description", "evidence", "recommendation"):
            if not isinstance(v[k], str) or not v[k].strip():
                errs.append(f"{p}.{k} must be a non-empty string")
    return errs


def write_report(path: str | Path, report: dict) -> None:
    """Atomically write the report (temp file + rename) after validating it."""
    problems = validate_report(report)
    if problems:
        raise ValueError("refusing to write invalid report: " + "; ".join(problems))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
