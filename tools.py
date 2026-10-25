"""Read-only repository inspection tools exposed to the LLM.

Safety properties
-----------------
* Nothing here writes to the target repository or executes its code.
* Paths are validated: relative only, no ``..``, no symlinks, no ``.git`` internals.
* All output is secret-redacted and size-capped before the model sees it.
* ``run_ruff`` is static analysis only and runs with ``--isolated`` so repo config
  (which could load plugins) is ignored.  ``run_pytest`` is intentionally NOT
  implemented: executing a test-suite means executing untrusted code and this
  project does not ship a container/VM sandbox.

The class also records which file lines the agent has actually *seen*; the findings
validator uses that to reject citations of lines the agent never looked at.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from redaction import LineRedactor, detect_secret_kinds, redact
from repo_access import clean_env

log = logging.getLogger("devops_agent.tools")

MAX_OUTPUT_CHARS = 14_000
MAX_FILE_BYTES = 1_000_000
DEFAULT_READ_LINES = 250
MAX_READ_LINES = 400
MAX_LINE_CHARS = 400
GIT_TIMEOUT = 30

_NESTED_QUANTIFIER = re.compile(r"\((?:[^()\\]|\\.)*[+*](?:[^()\\]|\\.)*\)[+*{]")


class ToolError(Exception):
    """A recoverable problem the model should see (bad path, bad argument...)."""


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


def _as_int(value: Any, name: str, default: int | None = None, lo: int = 1, hi: int = 10_000) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ToolError(f"'{name}' must be an integer")
    try:
        iv = int(value)
    except (TypeError, ValueError):
        raise ToolError(f"'{name}' must be an integer") from None
    return max(lo, min(hi, iv))


def _glob_match(rel: str, pattern: str) -> bool:
    p = pattern.lower().lstrip("./")
    r = rel.lower()
    return fnmatch.fnmatch(r, p) or fnmatch.fnmatch(os.path.basename(r), p) or fnmatch.fnmatch(r, "*/" + p)


# --------------------------------------------------------------------------- workflows
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_TEST_CMD_RE = re.compile(
    r"\b(pytest|py\.test|tox|nox|unittest|nosetests|npm (?:run )?test|yarn (?:run )?test|pnpm (?:run )?test|"
    r"go test|cargo test|dotnet test|rspec|phpunit|make (?:test|check)|ctest|jest|vitest|mocha|"
    r"mvnw? [^\n]*\btest\b|gradlew? [^\n]*\btest\b)",
    re.I,
)
_LINT_CMD_RE = re.compile(
    r"\b(ruff|flake8|pylint|black|isort|mypy|pyright|eslint|prettier|golangci-lint|go vet|"
    r"cargo clippy|cargo fmt|rubocop|pre-commit|checkstyle|spotless|make lint)\b",
    re.I,
)
_DEPLOY_RE = re.compile(
    r"\b(deploy|twine upload|npm publish|docker push|helm upgrade|kubectl apply|terraform apply|"
    r"gh release|pypi-publish|cargo publish|aws s3 sync)\b",
    re.I,
)
_INJECTION_RE = re.compile(
    r"\$\{\{\s*(?:github\.event\.(?:issue\.(?:title|body)|pull_request\.(?:title|body|head\.(?:ref|label))|"
    r"comment\.body|review\.body|review_comment\.body|head_commit\.message|"
    r"commits[^}]*\.message|pages[^}]*\.page_name)|github\.head_ref)\s*\}\}",
    re.I,
)
_CURL_PIPE_RE = re.compile(r"(curl|wget)[^|\n]*\|\s*(sudo\s+)?(ba|z)?sh\b", re.I)


def _classify_uses(uses: str) -> str:
    if uses.startswith("./"):
        return "local"
    if uses.startswith("docker://"):
        return "docker"
    if "@" not in uses:
        return "unversioned"
    ref = uses.rsplit("@", 1)[1]
    if _SHA_RE.match(ref):
        return "commit-sha"
    if re.match(r"^v?\d+(\.\d+){0,2}([\-+].*)?$", ref):
        return "version-tag"
    return "branch-or-other"


def _find_line(lines: list[str], needle: str, start: int = 0) -> int | None:
    needle = needle.strip()
    if not needle:
        return None
    for i in range(start, len(lines)):
        if needle in lines[i]:
            return i + 1
    return None


def summarize_workflow(text: str) -> tuple[dict, set[int]]:
    """Parse a GitHub Actions workflow into neutral facts.

    Returns ``(summary, lines_shown)``; the observations are *facts*, never verdicts.
    """
    import yaml  # local import keeps module import cheap

    lines = text.splitlines()
    shown: set[int] = set()
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        return {"parse_ok": False, "error": f"YAML parse error: {str(e).splitlines()[0][:200]}",
                "line": (mark.line + 1) if mark else None}, ({mark.line + 1} if mark else set())
    if not isinstance(data, dict):
        return {"parse_ok": False, "error": "workflow is not a YAML mapping"}, set()

    obs: list[str] = []

    def note(msg: str, line: int | None = None) -> None:
        if line:
            shown.add(line)
            msg = f"{msg} (line {line})"
        obs.append(msg)

    raw_on = data.get("on", data.get(True))  # PyYAML parses bare `on:` as boolean True
    triggers: dict[str, Any] = {}
    if isinstance(raw_on, str):
        triggers = {raw_on: {}}
    elif isinstance(raw_on, list):
        triggers = {str(e): {} for e in raw_on}
    elif isinstance(raw_on, dict):
        for k, v in raw_on.items():
            if isinstance(v, dict):
                triggers[str(k)] = {fk: fv for fk, fv in v.items() if fk != "inputs"}
            elif v is None:
                triggers[str(k)] = {}
            else:
                triggers[str(k)] = {"value": v}
    else:
        note("no `on:` trigger section found")

    ci_events = {"push", "pull_request", "pull_request_target", "release", "workflow_run", "workflow_call", "merge_group"}
    if triggers and not (set(triggers) & ci_events):
        note("workflow does not run on push, pull_request or release events (only: " + ", ".join(triggers) + ")")

    top_perm = data.get("permissions")
    if top_perm == "write-all":
        note("top-level permissions is write-all", _find_line(lines, "write-all"))

    has_prt = "pull_request_target" in triggers
    if has_prt:
        note("uses the pull_request_target trigger (runs with base-repo token/secrets)",
             _find_line(lines, "pull_request_target"))

    jobs = data.get("jobs") if isinstance(data.get("jobs"), dict) else {}
    if not jobs:
        note("no jobs defined")

    tests: list[str] = []
    lints: list[str] = []
    deploys: list[str] = []
    jobs_out: dict[str, Any] = {}
    cursor = 0
    for jname, job in jobs.items():
        if not isinstance(job, dict):
            continue
        jline = _find_line(lines, f"{jname}:", cursor)
        if jline:
            cursor = jline
        j: dict[str, Any] = {"line": jline}
        for key in ("runs-on", "needs", "if", "permissions", "environment", "timeout-minutes", "uses"):
            if key in job:
                j[key] = job[key]
        if job.get("continue-on-error") is True:
            j["continue-on-error"] = True
        runs_on = job.get("runs-on")
        if "self-hosted" in json.dumps(runs_on):
            note(f"job '{jname}' uses a self-hosted runner", jline)
        if top_perm is None and "permissions" not in job and "uses" not in job:
            note(f"job '{jname}' has no explicit permissions and there is no top-level permissions block", jline)
        if job.get("secrets") == "inherit":
            note(f"job '{jname}' passes `secrets: inherit` to a reusable workflow", jline)
        if str(job.get("if", "")).strip().lower() in {"false", "${{ false }}"}:
            note(f"job '{jname}' is disabled with `if: false`", jline)
        steps_out = []
        steps = job.get("steps") if isinstance(job.get("steps"), list) else []
        scur = jline or 0
        for idx, step in enumerate(steps, 1):
            if not isinstance(step, dict):
                continue
            s: dict[str, Any] = {"n": idx}
            if "name" in step:
                s["name"] = step["name"]
            uses = step.get("uses")
            run = step.get("run")
            sline = None
            if isinstance(uses, str):
                sline = _find_line(lines, uses, scur)
                s["uses"] = uses
                s["ref_kind"] = _classify_uses(uses)
                if s["ref_kind"] == "branch-or-other" and not uses.startswith("./"):
                    note(f"step uses `{uses}` which is a branch/mutable ref", sline)
                if uses.startswith("actions/checkout") and isinstance(step.get("with"), dict):
                    ref = str(step["with"].get("ref", ""))
                    if has_prt and re.search(r"pull_request\.head|head_ref|head\.sha", ref):
                        note(f"job '{jname}' checks out untrusted PR head code under pull_request_target", sline)
                if isinstance(step.get("with"), dict):
                    s["with_keys"] = sorted(map(str, step["with"].keys()))
            if isinstance(run, str):
                first = next((ln for ln in run.splitlines() if ln.strip()), "")
                sline = _find_line(lines, first, scur)
                s["run"] = redact(run.strip())[:300]
                if _INJECTION_RE.search(run):
                    note(f"job '{jname}' step {idx} interpolates attacker-controllable event data directly into a shell script", sline)
                if _CURL_PIPE_RE.search(run):
                    note(f"job '{jname}' step {idx} pipes a download straight into a shell", sline)
                is_test, is_lint = bool(_TEST_CMD_RE.search(run)), bool(_LINT_CMD_RE.search(run))
                if is_test:
                    tests.append(f"{jname}#{idx}: {_TEST_CMD_RE.search(run).group(0)}")
                if is_lint:
                    lints.append(f"{jname}#{idx}: {_LINT_CMD_RE.search(run).group(0)}")
                if _DEPLOY_RE.search(run):
                    deploys.append(f"{jname}#{idx}: {_DEPLOY_RE.search(run).group(0)}")
                if (is_test or is_lint) and re.search(r"\|\|\s*(true|exit 0)\b", run):
                    note(f"job '{jname}' step {idx} runs a test/lint command but swallows failures with `|| true`", sline)
            if isinstance(uses, str) and _DEPLOY_RE.search(uses):
                deploys.append(f"{jname}#{idx}: {uses}")
            if step.get("continue-on-error") is True:
                s["continue-on-error"] = True
                blob = f"{step.get('name', '')} {run or ''} {uses or ''}"
                if _TEST_CMD_RE.search(blob) or _LINT_CMD_RE.search(blob):
                    note(f"job '{jname}' step {idx} is a test/lint step with continue-on-error: true", sline)
            if "if" in step:
                s["if"] = str(step["if"])[:120]
                if str(step["if"]).strip().lower() in {"false", "${{ false }}"}:
                    note(f"job '{jname}' step {idx} is disabled with `if: false`", sline)
            if sline:
                s["line"] = sline
                shown.add(sline)
                scur = sline
            steps_out.append(s)
        j["steps"] = steps_out
        jobs_out[str(jname)] = j

    summary = {
        "parse_ok": True,
        "name": data.get("name"),
        "triggers": triggers,
        "top_level_permissions": top_perm if top_perm is not None else "not set",
        "jobs": jobs_out,
        "test_commands_found": tests,
        "lint_commands_found": lints,
        "deploy_like_commands_found": deploys,
        "observations": obs,
    }
    return summary, shown


# --------------------------------------------------------------------------- tools
class RepoTools:
    """Tool implementations bound to one checked-out repository."""

    def __init__(self, root: str | Path, *, max_output_chars: int = MAX_OUTPUT_CHARS,
                 search_timeout: float = 15.0) -> None:
        self.root = Path(root).resolve()
        self.max_output_chars = max_output_chars
        self.search_timeout = search_timeout
        self.has_git = (self.root / ".git").exists()
        self.seen_lines: dict[str, set[int]] = {}
        self.listed_files = False
        self._files: list[str] | None = None
        self._handlers: dict[str, Callable[..., str]] = {
            "list_files": self.list_files,
            "read_file": self.read_file,
            "search_code": self.search_code,
            "find_files": self.find_files,
            "git_log": self.git_log,
            "inspect_workflow": self.inspect_workflow,
            "scan_secrets": self.scan_secrets,
            "run_ruff": self.run_ruff,
        }

    # ------------------------------------------------------------ plumbing
    @property
    def tool_names(self) -> list[str]:
        return list(self._handlers)

    def execute(self, name: str, arguments: Any) -> ToolResult:
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(f"Error: unknown tool '{name}'. Available tools: {', '.join(self._handlers)}", True)
        if not isinstance(arguments, dict):
            return ToolResult("Error: tool arguments must be a JSON object", True)
        try:
            out = handler(**arguments)
        except ToolError as e:
            return ToolResult(f"Error: {e}", True)
        except TypeError as e:
            return ToolResult(f"Error: invalid arguments for {name}: {e}", True)
        except subprocess.TimeoutExpired:
            return ToolResult(f"Error: {name} timed out", True)
        except Exception as e:  # tools must never crash the loop
            log.warning("tool %s failed: %s", name, type(e).__name__)
            return ToolResult(f"Error: {name} failed unexpectedly ({type(e).__name__}: {redact(str(e))[:200]})", True)
        return ToolResult(self._cap(redact(out)), False)

    def _cap(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        return text[: self.max_output_chars] + "\n…[output truncated; narrow your query]"

    def _all_files(self) -> list[str]:
        if self._files is None:
            out: list[str] = []
            for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
                dirnames[:] = [d for d in dirnames if d != ".git"]
                for f in filenames:
                    full = os.path.join(dirpath, f)
                    if os.path.islink(full):
                        continue
                    out.append(os.path.relpath(full, self.root).replace(os.sep, "/"))
            out.sort()
            self._files = out
        return self._files

    def normalize(self, rel: Any) -> str:
        if not isinstance(rel, str) or not rel.strip():
            raise ToolError("path must be a non-empty string")
        s = rel.strip().replace("\\", "/")
        if "\x00" in s:
            raise ToolError("invalid path")
        while s.startswith("./"):
            s = s[2:]
        p = PurePosixPath(s)
        if p.is_absolute() or ".." in p.parts:
            raise ToolError("path must be relative to the repository root and must not contain '..'")
        if p.parts and p.parts[0] == ".git":
            raise ToolError(".git internals are not accessible; use git_log for history")
        return "/".join(p.parts)

    def _resolve(self, rel: str) -> tuple[str, Path]:
        norm = self.normalize(rel)
        cur = self.root
        for part in norm.split("/"):
            cur = cur / part
            if cur.is_symlink():
                raise ToolError("symlinks are not followed")
        resolved = cur.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ToolError("path escapes the repository")
        return norm, cur

    def exists(self, rel: str) -> bool:
        try:
            _, full = self._resolve(rel)
        except ToolError:
            return False
        return full.exists()

    def mark_seen(self, rel: str, lines: Any) -> None:
        self.seen_lines.setdefault(rel, set()).update(lines)

    def line_seen(self, rel: str, line: int) -> bool:
        return line in self.seen_lines.get(rel, set())

    def matching_paths(self, pattern: str) -> list[str]:
        """Repo paths matching a glob (or a ``dir/`` prefix); used to verify absence claims."""
        pat = pattern.strip().replace("\\", "/")
        if pat.endswith("/"):
            return [f for f in self._all_files() if f.lower().startswith(pat.lower().lstrip("./"))][:5]
        return [f for f in self._all_files() if _glob_match(f, pat)][:5]

    @staticmethod
    def _is_binary(path: Path) -> bool:
        try:
            with open(path, "rb") as fh:
                return b"\0" in fh.read(8192)
        except OSError:
            return True

    def _iter_text_files(self, path_glob: str | None):
        for rel in self._all_files():
            if path_glob and not _glob_match(rel, path_glob):
                continue
            full = self.root / rel
            try:
                if full.stat().st_size > MAX_FILE_BYTES or self._is_binary(full):
                    continue
            except OSError:
                continue
            yield rel, full

    # ------------------------------------------------------------ tools
    def list_files(self, path: str = "", max_entries: int = 250) -> str:
        """Discover repository structure (tracked files only; .git is hidden)."""
        max_entries = _as_int(max_entries, "max_entries", 250, 20, 1000)
        prefix = ""
        if path not in ("", ".", "/", None):
            prefix = self.normalize(path).rstrip("/") + "/"
        files = [f for f in self._all_files() if f.startswith(prefix)]
        self.listed_files = True
        if not files:
            return f"No files under '{prefix or '.'}'" + ("" if prefix else " (the repository checkout is empty)")
        rels = [f[len(prefix):] for f in files]
        ext = Counter(os.path.splitext(r)[1].lower() or "(none)" for r in rels)
        head = [f"Directory: {prefix or '.'}  |  {len(rels)} files  |  top extensions: "
                + ", ".join(f"{e}={n}" for e, n in ext.most_common(8))]

        def entries_at(depth: int) -> dict[str, tuple[bool, int]]:
            out: dict[str, tuple[bool, int]] = {}
            for r in rels:
                parts = r.split("/")
                if len(parts) <= depth:
                    out[r] = (False, 1)
                else:
                    key = "/".join(parts[:depth]) + "/"
                    n = out.get(key, (True, 0))[1]
                    out[key] = (True, n + 1)
            return out

        chosen = entries_at(1)
        for d in range(2, 12):
            cand = entries_at(d)
            if len(cand) > max_entries:
                break
            chosen = cand
            if all(not is_dir for is_dir, _ in cand.values()):
                break
        for name in sorted(chosen):
            is_dir, n = chosen[name]
            if is_dir:
                head.append(f"{prefix}{name}  ({n} files)")
            else:
                try:
                    size = (self.root / prefix / name).stat().st_size
                except OSError:
                    size = -1
                head.append(f"{prefix}{name}  [{size} B]")
        if any(is_dir and n > 1 for is_dir, n in chosen.values()):
            head.append("(directories with counts are collapsed; call list_files(path='<dir>/') to expand)")
        return "\n".join(head)

    def find_files(self, pattern: str, max_results: int = 100) -> str:
        """Find files by glob (e.g. 'requirements*.txt', '*.lock', '.github/workflows/*')."""
        if not isinstance(pattern, str) or not pattern.strip():
            raise ToolError("pattern must be a non-empty glob string")
        max_results = _as_int(max_results, "max_results", 100, 1, 300)
        self.listed_files = True
        hits = [f for f in self._all_files() if _glob_match(f, pattern.strip())]
        if not hits:
            return f"No files match '{pattern}' (searched {len(self._all_files())} files)."
        shown = hits[:max_results]
        extra = f"\n…and {len(hits) - len(shown)} more" if len(hits) > len(shown) else ""
        return f"{len(hits)} file(s) match '{pattern}':\n" + "\n".join(shown) + extra

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read a text file with line numbers (default first 250 lines)."""
        rel, full = self._resolve(path)
        if not full.exists():
            raise ToolError(f"file not found: {rel} (use list_files or find_files to discover paths)")
        if full.is_dir():
            raise ToolError(f"{rel} is a directory; use list_files(path='{rel}/')")
        size = full.stat().st_size
        if self._is_binary(full):
            return f"{rel}: binary file, {size} bytes (content not shown)"
        start = _as_int(start_line, "start_line", 1, 1, 10**9)
        end = _as_int(end_line, "end_line", start + DEFAULT_READ_LINES - 1, 1, 10**9)
        if end < start:
            raise ToolError("end_line must be >= start_line")
        end = min(end, start + MAX_READ_LINES - 1)
        budget = self.max_output_chars - 400
        redactor = LineRedactor()
        rows: list[str] = []
        used = 0
        last = start - 1
        more = False
        total_seen = 0
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            for i, raw in enumerate(fh, 1):
                total_seen = i
                text, _ = redactor.redact_line(raw.rstrip("\r\n"))  # state advances before `start`
                if i < start:
                    continue
                if i > end:
                    more = True
                    break
                if len(text) > MAX_LINE_CHARS:
                    text = text[:MAX_LINE_CHARS] + " …[line truncated]"
                row = f"{i:>5} | {text}"
                if used + len(row) > budget:
                    more = True
                    break
                rows.append(row)
                used += len(row) + 1
                last = i
        if not rows:
            if size == 0:
                return f"{rel}: empty file"
            return f"{rel}: file has only {total_seen} lines (requested start_line={start})"
        self.mark_seen(rel, range(start, last + 1))
        out = [f"=== {rel} (lines {start}-{last}" + (", more available" if more else ", end of file") + ") ===", *rows]
        if more:
            out.append(f"[call read_file(path='{rel}', start_line={last + 1}) to continue]")
        return "\n".join(out)

    def search_code(self, pattern: str, path_glob: str | None = None, case_sensitive: bool = False,
                    max_results: int = 40) -> str:
        """Regex search across text files; returns path:line: text (secret values redacted)."""
        if not isinstance(pattern, str) or not pattern:
            raise ToolError("pattern must be a non-empty string")
        max_results = _as_int(max_results, "max_results", 40, 1, 100)
        if _NESTED_QUANTIFIER.search(pattern):
            raise ToolError("pattern rejected: nested quantifiers risk catastrophic backtracking; simplify it")
        flags = 0 if case_sensitive else re.IGNORECASE
        note = ""
        try:
            rx = re.compile(pattern, flags)
        except re.error as e:
            rx = re.compile(re.escape(pattern), flags)
            note = f"(note: not a valid regex ({e}); searched as literal text)\n"
        deadline = time.monotonic() + self.search_timeout
        hits: list[tuple[str, int, str]] = []
        scanned = 0
        timed_out = False
        files_with_hits: set[str] = set()
        for rel, full in self._iter_text_files(path_glob):
            if time.monotonic() > deadline:
                timed_out = True
                break
            scanned += 1
            redactor = LineRedactor()
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as fh:
                    for i, raw in enumerate(fh, 1):
                        line = raw.rstrip("\r\n")
                        safe, is_key_body = redactor.redact_line(line)
                        if is_key_body:
                            continue
                        if rx.search(line[:2000]):
                            hits.append((rel, i, safe[:300]))
                            files_with_hits.add(rel)
                            if len(hits) >= max_results:
                                break
                        if i % 5000 == 0 and time.monotonic() > deadline:
                            timed_out = True
                            break
            except OSError:
                continue
            if len(hits) >= max_results or timed_out:
                break
        if not hits:
            return note + f"No matches for /{pattern}/ (scanned {scanned} text files)" + (" - search timed out" if timed_out else "")
        for rel, i, _ in hits:
            self.mark_seen(rel, [i])
        body = "\n".join(f"{rel}:{i}: {txt.strip()}" for rel, i, txt in hits)
        tail = ""
        if len(hits) >= max_results:
            tail = f"\n[stopped at {max_results} matches; narrow the pattern or use path_glob]"
        if timed_out:
            tail += "\n[search timed out; results incomplete]"
        return note + f"{len(hits)} match(es) in {len(files_with_hits)} file(s) (scanned {scanned} files):\n{body}{tail}"

    def _git(self, args: list[str]) -> str:
        cmd = ["git", "--no-pager", "--no-optional-locks", "-c", "core.hooksPath=/dev/null",
               "-c", "log.showSignature=false", *args]
        res = subprocess.run(cmd, cwd=self.root, env=clean_env(), capture_output=True, text=True,
                             errors="replace", timeout=GIT_TIMEOUT)
        if res.returncode != 0:
            raise ToolError(f"git failed: {(res.stderr or '').strip()[:200]}")
        return res.stdout

    def git_log(self, max_count: int = 20, path: str | None = None, include_files: bool = False) -> str:
        """Recent commit history (hash, date, author name, subject); optionally changed files."""
        if not self.has_git or not shutil.which("git"):
            return "Git history is not available for this checkout (no .git directory or git missing)."
        n = _as_int(max_count, "max_count", 20, 1, 100)
        args = ["log", f"--max-count={n}", "--date=short",
                "--pretty=format:%x1e%h%x1f%ad%x1f%an%x1f%s"]
        if include_files:
            args.append("--name-status")
        if path:
            args += ["--", self.normalize(path)]
        raw = self._git(args)
        try:
            shallow = self._git(["rev-parse", "--is-shallow-repository"]).strip() == "true"
            total = self._git(["rev-list", "--count", "HEAD"]).strip()
        except ToolError:
            shallow, total = False, "?"
        out = []
        for chunk in raw.split("\x1e"):
            if not chunk.strip():
                continue
            head, _, files = chunk.partition("\n")
            parts = head.split("\x1f")
            if len(parts) < 4:
                continue
            out.append(f"{parts[0]} {parts[1]} {parts[2]}: {parts[3]}")
            if include_files:
                fl = [ln for ln in files.splitlines() if ln.strip()]
                out.extend("    " + ln for ln in fl[:15])
                if len(fl) > 15:
                    out.append(f"    …(+{len(fl) - 15} more files)")
        if not out:
            return "No commits found" + (f" for path '{path}'" if path else "")
        header = f"{len(out) if not include_files else sum(1 for o in out if not o.startswith(' '))} commit(s) shown; " \
                 f"{total} reachable in this clone" + (" (SHALLOW clone: older history not available)" if shallow else "")
        return header + "\n" + "\n".join(out)

    def inspect_workflow(self, path: str) -> str:
        """Parse a GitHub Actions workflow YAML into structured facts and observations."""
        rel, full = self._resolve(path)
        if not full.is_file():
            raise ToolError(f"workflow file not found: {rel}")
        if full.stat().st_size > MAX_FILE_BYTES:
            raise ToolError("workflow file too large")
        text = full.read_text(encoding="utf-8", errors="replace")
        summary, shown = summarize_workflow(text)
        self.mark_seen(rel, shown)
        return f"Workflow summary for {rel} (facts only; use read_file for exact lines):\n" + json.dumps(summary, indent=1, default=str)

    def scan_secrets(self, path_glob: str | None = None, max_results: int = 60) -> str:
        """Locate probable hard-coded secrets. Reports file:line and KIND only - never values."""
        max_results = _as_int(max_results, "max_results", 60, 1, 200)
        found: list[str] = []
        positions: list[tuple[str, int]] = []
        scanned = 0
        for rel, full in self._iter_text_files(path_glob):
            scanned += 1
            low = rel.lower()
            tags = []
            if re.search(r"(^|/)(tests?|spec|__tests__|fixtures?)(/|$)|test_|_test\.", low):
                tags.append("test-code")
            if re.search(r"example|sample|template|\.dist$", low):
                tags.append("example/template")
            if low.endswith((".md", ".rst", ".txt")) or "/docs/" in low or low.startswith("docs/"):
                tags.append("docs")
            if re.search(r"(^|/)\.env(\.|$)", low) and "example" not in low and "sample" not in low:
                tags.append("env-file")
            redactor = LineRedactor()
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as fh:
                    for i, raw in enumerate(fh, 1):
                        line = raw.rstrip("\r\n")
                        _, key_body = redactor.redact_line(line)
                        kinds = ["private_key_body"] if key_body else detect_secret_kinds(line[:2000])
                        if redactor.in_block and not key_body:
                            kinds = kinds or ["private_key"]
                        if kinds:
                            positions.append((rel, i))
                            found.append(f"{rel}:{i}  {','.join(sorted(set(kinds)))}" + (f"  [{', '.join(tags)}]" if tags else ""))
                            if len(found) >= max_results:
                                break
            except OSError:
                continue
            if len(found) >= max_results:
                break
        for rel, i in positions:
            self.mark_seen(rel, [i])
        if not found:
            return f"No probable hard-coded secrets found in {scanned} text files (pattern-based; not proof of absence)."
        more = "\n[result limit reached]" if len(found) >= max_results else ""
        return (f"{len(found)} probable secret location(s) in {scanned} files (values withheld). "
                "Context tags help you judge severity; verify with read_file before reporting:\n"
                + "\n".join(found) + more)

    def run_ruff(self, max_results: int = 25) -> str:
        """Static Python lint with ruff in --isolated mode (ignores repo config; informational)."""
        max_results = _as_int(max_results, "max_results", 25, 1, 100)
        py_files = [f for f in self._all_files() if f.endswith(".py")]
        if not py_files:
            return "No Python files in this repository."
        exe = shutil.which("ruff")
        cmd = [exe] if exe else [sys.executable, "-m", "ruff"]
        cmd += ["check", "--isolated", "--no-cache", "--output-format=json", "--exit-zero",
                "--select", "E9,F63,F7,F82,F401,F811,F841", "."]
        try:
            res = subprocess.run(cmd, cwd=self.root, env=clean_env(), capture_output=True, text=True,
                                 errors="replace", timeout=90)
        except FileNotFoundError:
            return "ruff is not installed in the audit environment (pip install ruff)."
        if res.returncode not in (0, 1) or not res.stdout.strip():
            msg = (res.stderr or "").strip()[:200]
            if "No module named ruff" in msg:
                return "ruff is not installed in the audit environment (pip install ruff)."
            raise ToolError(f"ruff failed: {msg}")
        try:
            diags = json.loads(res.stdout)
        except ValueError:
            raise ToolError("could not parse ruff output") from None
        counts = Counter(d.get("code") for d in diags)
        lines = [f"ruff (isolated defaults, rules E9,F63,F7,F82,F401,F811,F841) on {len(py_files)} Python files: "
                 f"{len(diags)} diagnostic(s). NOTE: this ignores the repo's own lint config - informational only; "
                 "do not report unused imports alone as a violation.",
                 "By rule: " + ", ".join(f"{c}={n}" for c, n in counts.most_common(8))]
        for d in diags[:max_results]:
            fname = os.path.relpath(d.get("filename", ""), self.root).replace(os.sep, "/")
            row = (d.get("location") or {}).get("row")
            lines.append(f"{fname}:{row}: {d.get('code')} {str(d.get('message'))[:120]}")
            if isinstance(row, int):
                self.mark_seen(fname, [row])
        if len(diags) > max_results:
            lines.append(f"…and {len(diags) - max_results} more")
        return "\n".join(lines)


# --------------------------------------------------------------------------- schemas
def tool_definitions() -> list[dict]:
    """Anthropic-style tool schemas for the inspection tools."""
    obj = lambda props, req=(): {"type": "object", "properties": props, "required": list(req)}  # noqa: E731
    return [
        {"name": "list_files",
         "description": "List repository files and structure (large directories are collapsed with counts). "
                        "Call with path='<dir>/' to expand a directory.",
         "input_schema": obj({"path": {"type": "string", "description": "Directory to list; default repo root"},
                              "max_entries": {"type": "integer"}})},
        {"name": "find_files",
         "description": "Find files by glob, e.g. 'requirements*.txt', '*.lock', '.github/workflows/*', 'README*'. "
                        "Use this to check whether a file or an equivalent exists before claiming it is missing.",
         "input_schema": obj({"pattern": {"type": "string"}, "max_results": {"type": "integer"}}, ["pattern"])},
        {"name": "read_file",
         "description": "Read a text file with line numbers (250 lines by default; use start_line/end_line to page). "
                        "Secret values are redacted.",
         "input_schema": obj({"path": {"type": "string"}, "start_line": {"type": "integer"},
                              "end_line": {"type": "integer"}}, ["path"])},
        {"name": "search_code",
         "description": "Regex search (case-insensitive by default) across text files. Returns path:line: text. "
                        "Optionally restrict with path_glob such as '*.py' or '.github/*'.",
         "input_schema": obj({"pattern": {"type": "string"}, "path_glob": {"type": "string"},
                              "case_sensitive": {"type": "boolean"}, "max_results": {"type": "integer"}}, ["pattern"])},
        {"name": "git_log",
         "description": "Show commit history (hash, date, author, subject). include_files=true adds changed files per "
                        "commit (useful to spot committed-then-deleted secrets or generated files). Clone may be shallow.",
         "input_schema": obj({"max_count": {"type": "integer"}, "path": {"type": "string"},
                              "include_files": {"type": "boolean"}})},
        {"name": "inspect_workflow",
         "description": "Parse a GitHub Actions workflow (.github/workflows/*.yml) into facts: triggers, permissions, "
                        "jobs/steps, action ref kinds, test/lint/deploy commands found, and risk observations.",
         "input_schema": obj({"path": {"type": "string"}}, ["path"])},
        {"name": "scan_secrets",
         "description": "Pattern-scan for probable hard-coded credentials. Returns file:line and secret KIND with context "
                        "tags (test-code, example/template, docs, env-file). Never returns values.",
         "input_schema": obj({"path_glob": {"type": "string"}, "max_results": {"type": "integer"}})},
        {"name": "run_ruff",
         "description": "Run ruff static analysis in isolated mode on Python files (informational; ignores the repo's "
                        "own config; does not execute repo code).",
         "input_schema": obj({"max_results": {"type": "integer"}})},
    ]
