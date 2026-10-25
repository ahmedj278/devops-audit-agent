"""Secret redaction used for tool output, logs and the final report.

Design rules:
  * The LLM never sees a secret value: tool output is redacted first.
  * Placeholders / references (``${TOKEN}``, ``os.environ[...]``, ``your_key_here``)
    are deliberately left visible so the model can tell them apart from real values.
  * ``detect_secret_kinds`` reports *kinds* only, never values.
"""

from __future__ import annotations

import logging
import re
from typing import Any

_KNOWN_SECRETS: set[str] = set()


def register_secret(value: str | None) -> None:
    """Register an exact secret (e.g. our own API key) to be scrubbed everywhere."""
    if value and len(value) >= 8:
        _KNOWN_SECRETS.add(value)


_SIMPLE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "private_key",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----[\s\S]*?"
            r"(?:-----END [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----|$)"
        ),
    ),
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    (
        "github_token",
        re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    ),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("stripe_key", re.compile(r"\b[sr]k_(?:live|test)_[0-9A-Za-z]{16,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{32,}\b")),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
    ),
]

_URL_CRED = re.compile(
    r"(?P<pre>\b[a-z][a-z0-9+.\-]*://[^\s:/@\"']+:)(?P<pw>[^\s@/\"']+)(?P<post>@)", re.I
)

_GENERIC = re.compile(
    r"(?P<name>[A-Za-z0-9_.\-]*(?:password|passwd|pwd|secret|token|api[_\-]?key|apikey|"
    r"access[_\-]?key|private[_\-]?key|auth[_\-]?key|credential)[A-Za-z0-9_.\-]*[\"']?)"
    r"(?P<sep>\s*[:=]\s*)(?P<q>[\"']?)(?P<val>[^\s\"',;#]{6,})(?P=q)",
    re.I,
)

_PLACEHOLDER_WORDS = {
    "password", "passwd", "secret", "token", "apikey", "api_key", "changeme", "none",
    "null", "true", "false", "required", "empty", "redacted", "string", "example",
    "test", "testing", "dummy", "placeholder", "undefined",
}
_PLACEHOLDER_SUBSTR = (
    "your", "example", "changeme", "change_me", "placeholder", "xxxx", "****", "<", ">",
    "${", "{{", "%(", "$(", "dummy", "sample", "todo", "fixme", "replace", "insert",
    "enter", "here", "redacted",
)
_REF_PREFIXES = ("os.", "env", "getenv", "process.", "secrets.", "config", "settings", "$", "http://", "https://")


def looks_like_placeholder(value: str) -> bool:
    v = value.strip().lower()
    if v in _PLACEHOLDER_WORDS:
        return True
    if len(set(v)) <= 2:  # "aaaaaa", "******"
        return True
    return any(s in v for s in _PLACEHOLDER_SUBSTR)


def looks_like_reference(value: str) -> bool:
    v = value.strip()
    if "(" in v or "[" in v:
        return True
    if v.lower().startswith(_REF_PREFIXES):
        return True
    return bool(re.fullmatch(r"[A-Za-z_][\w]*(\.[A-Za-z_]\w*)+", v))


def _generic_is_secret(val: str) -> bool:
    v = val.strip()
    if looks_like_placeholder(v) or looks_like_reference(v):
        return False
    if not re.search(r"[A-Za-z]", v):
        return False
    return bool(re.search(r"[\d\W_]", v)) or len(v) >= 10


def _generic_sub(m: re.Match[str]) -> str:
    if not _generic_is_secret(m.group("val")):
        return m.group(0)
    q = m.group("q")
    return f"{m.group('name')}{m.group('sep')}{q}[REDACTED:generic_secret]{q}"


def _url_sub(m: re.Match[str]) -> str:
    if looks_like_placeholder(m.group("pw")) or looks_like_reference(m.group("pw")):
        return m.group(0)
    return f"{m.group('pre')}[REDACTED:url_password]{m.group('post')}"


def redact(text: str) -> str:
    """Replace secret values in *text* with ``[REDACTED:<kind>]`` markers."""
    if not text:
        return text
    for s in _KNOWN_SECRETS:
        text = text.replace(s, "[REDACTED:env_secret]")
    for kind, rx in _SIMPLE_PATTERNS:
        text = rx.sub(f"[REDACTED:{kind}]", text)
    text = _URL_CRED.sub(_url_sub, text)
    text = _GENERIC.sub(_generic_sub, text)
    return text


def redact_obj(obj: Any) -> Any:
    """Recursively redact every string in a JSON-like structure."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, list):
        return [redact_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    return obj


def detect_secret_kinds(line: str) -> list[str]:
    """Return the *kinds* of probable real secrets on a line (never the values)."""
    kinds: list[str] = []
    for kind, rx in _SIMPLE_PATTERNS:
        m = rx.search(line)
        if m and "EXAMPLE" not in m.group(0):
            kinds.append(kind)
    for m in _URL_CRED.finditer(line):
        if not (looks_like_placeholder(m.group("pw")) or looks_like_reference(m.group("pw"))):
            kinds.append("url_password")
            break
    for m in _GENERIC.finditer(line):
        if _generic_is_secret(m.group("val")):
            kinds.append("generic_secret_assignment")
            break
    return kinds


class LineRedactor:
    """Stateful per-file redactor that also hides multi-line private-key bodies."""

    _BEGIN = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----")
    _END = re.compile(r"-----END [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----")

    def __init__(self) -> None:
        self.in_block = False

    def redact_line(self, line: str) -> tuple[str, bool]:
        """Return ``(safe_text, is_key_body_line)``."""
        if self.in_block:
            if self._END.search(line):
                self.in_block = False
            return "[REDACTED:private_key_body]", True
        if self._BEGIN.search(line) and not self._END.search(line):
            self.in_block = True
        return redact(line), False


class RedactingFilter(logging.Filter):
    """Logging filter that scrubs secrets from every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact(record.getMessage())
            record.args = ()
        except Exception:  # never let logging break the run
            pass
        return True
