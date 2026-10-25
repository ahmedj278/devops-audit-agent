#!/usr/bin/env python3
"""Regenerate ``sample_report.json`` WITHOUT an LLM API key.

IMPORTANT - what this is and is not
-----------------------------------
This runs the *real* pipeline (CLI ``agent.main`` -> tool loop -> tools -> findings validation ->
report writer) but the "model" is a small deterministic stand-in policy (``StandInModel`` below),
not a real LLM, and the target is a local fixture repository with planted defects rather than a
repository fetched from GitHub.  Everything in the report went through the same validation as a
real run (cited lines must have been shown by tools, absence claims are verified, secrets are
redacted), but the *choice* of what to investigate came from this script.

For a report produced by a real LLM against a real repository, run:

    python agent.py --repo https://github.com/<owner>/<repo> --output sample_report.json

Usage:  python scripts/generate_sample.py [output_path]
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent as cli  # noqa: E402
import repo_access  # noqa: E402
from tests.fakes import ScriptedLLM, all_results_text, call, reply  # noqa: E402

DEMO_URL = "https://github.com/example/demo-project"

# Assembled at runtime so no credential-shaped literal exists in this source file.
_FAKE_KEY = "AKIA" + "ZZZZ1111AAAA2222"

WORKFLOW = """\
name: CI
on: pull_request_target
jobs:
  checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}
      - name: Greet
        run: echo "Thanks for ${{ github.event.pull_request.title }}"
      - name: Tests
        run: pytest || true
"""

FILES = {
    "app.py": "import os\nimport sys\n\n\ndef main():\n    return 'ok'\n",
    "config.py": f'AWS_ACCESS_KEY_ID = "{_FAKE_KEY}"\nREGION = "eu-west-1"\nDATA_DIR = "/home/alice/projects/demo/data"\n',
    "requirements.txt": "flask\nrequests>=2\npytest==8.2.0\n",
    "tests/test_app.py": "from app import main\n\n\ndef test_main():\n    assert main() == 'ok'\n",
    ".github/workflows/ci.yml": WORKFLOW,
    "__pycache__/app.cpython-311.pyc": b"\x00\x00\x00\x00bytecode",
    ".DS_Store": b"\x00\x00\x00\x01Bud1",
}


def build_fixture(dest: Path) -> None:
    dest.mkdir(parents=True)

    def git(*a):
        subprocess.run(["git", "-c", "user.name=Demo", "-c", "user.email=demo@example.com",
                        "-c", "commit.gpgsign=false", *a], cwd=dest, check=True, capture_output=True)

    git("init", "-q")
    for rel, content in FILES.items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content if isinstance(content, bytes) else content.encode())
    git("add", "-A")
    git("commit", "-q", "-m", "Initial commit")


class StandInModel:
    """Deterministic, observation-driven stand-in for an LLM (NOT an LLM)."""

    def __call__(self, messages, n):
        seen = all_results_text(messages)
        used = [(b["name"], tuple(sorted(b["input"].items())))
                for m in messages if m["role"] == "assistant" for b in m["content"] if b.get("type") == "tool_use"]

        def did(name, **kw):
            return (name, tuple(sorted(kw.items()))) in used

        def once(*calls):
            fresh = [c for c in calls if not did(c.name, **c.input)]
            return fresh

        if n == 1:
            return reply(call("list_files"), call("find_files", pattern="README*"),
                         call("find_files", pattern=".gitignore"), call("find_files", pattern="requirements*"))
        # follow leads found in the listing / earlier results
        batch = []
        for wf in dict.fromkeys(re.findall(r"\.github/workflows/[\w.\-]+\.ya?ml", seen)):
            batch += once(call("inspect_workflow", path=wf))
        batch += once(call("scan_secrets"))
        if "requirements.txt" in seen:
            batch += once(call("read_file", path="requirements.txt"))
        batch += once(call("search_code", pattern=r"/home/|/Users/|[A-Z]:\\\\Users"))
        if batch:
            return reply(*batch)
        hits = re.findall(r"^(\S+?):(\d+)\s+\w+", seen, re.M)  # scan_secrets output lines
        reads = [c for c in (call("read_file", path=p) for p, _ in dict.fromkeys(hits))
                 if not did("read_file", path=c.input["path"])]
        if reads:
            return reply(*reads)
        if not did("report_violation_batch"):  # sentinel never used; reports are issued once below
            reported = "report_violation" in {u[0] for u in used}
            if not reported:
                return reply(*self._reports(seen))
        return reply(call("finish_audit"))

    @staticmethod
    def _reports(seen: str):
        out = []
        m = re.search(r"^(\S+?):(\d+)\s+aws_access_key_id", seen, re.M)
        if m:
            out.append(call(
                "report_violation", category="security", severity="critical", file=m.group(1), line=int(m.group(2)),
                description="Hardcoded AWS access key ID in source",
                evidence="scan_secrets and read_file show an AWS access key id assigned as a string literal at module level.",
                recommendation="Remove it from the repository, rotate the key, and load credentials from the environment or a secrets manager."))
        wf = ".github/workflows/ci.yml"
        m = re.search(r"checks out untrusted PR head code under pull_request_target \(line (\d+)\)", seen)
        if m:
            out.append(call(
                "report_violation", category="ci_cd", severity="critical", file=wf, line=int(m.group(1)),
                description="pull_request_target workflow checks out untrusted PR code",
                evidence="The workflow triggers on pull_request_target and actions/checkout is given the pull request head SHA, so fork code runs with the base repository's token and secrets.",
                recommendation="Trigger on pull_request instead, or split into an unprivileged build job and a privileged job that never checks out or executes PR code."))
        m = re.search(r"interpolates attacker-controllable event data directly into a shell script \(line (\d+)\)", seen)
        if m:
            out.append(call(
                "report_violation", category="security", severity="high", file=wf, line=int(m.group(1)),
                description="Pull request title interpolated into a shell script (script injection)",
                evidence="A run step expands the pull request title expression inside the script body, letting a crafted title execute commands.",
                recommendation="Pass the value through an environment variable and reference it as a quoted shell variable."))
        m = re.search(r"runs a test/lint command but swallows failures with `\|\| true` \(line (\d+)\)", seen)
        if m:
            out.append(call(
                "report_violation", category="testing", severity="high", file=wf, line=int(m.group(1)),
                description="CI test step cannot fail the build",
                evidence="The pytest step ends in '|| true', so test failures never fail the workflow.",
                recommendation="Remove '|| true' so a failing test suite blocks the merge."))
        if "No files match 'README*'" in seen:
            out.append(call(
                "report_violation", category="repository_hygiene", severity="low", file=None, line=None,
                description="Repository has no README",
                evidence="find_files('README*') matched nothing in the repository tree.",
                recommendation="Add a README.md with purpose, setup, and usage instructions.",
                missing_paths=["README*"]))
        if "No files match '.gitignore'" in seen and "__pycache__" in seen:
            out.append(call(
                "report_violation", category="repository_hygiene", severity="medium",
                file="__pycache__/app.cpython-311.pyc", line=None,
                description="Generated files committed and no .gitignore present",
                evidence="A compiled Python bytecode file and a macOS .DS_Store file are tracked, and no .gitignore exists to exclude them.",
                recommendation="Add a .gitignore (Python and OS templates) and remove the tracked artifacts with git rm --cached.",
                missing_paths=[".gitignore"]))
        m = re.search(r"^\s*(\d+) \| (flask)\s*$", seen, re.M)
        if m:
            out.append(call(
                "report_violation", category="dependency_management", severity="medium", file="requirements.txt",
                line=int(m.group(1)),
                description="Unpinned runtime dependency in requirements.txt",
                evidence="read_file shows 'flask' with no version constraint and 'requests' with only a lower bound ('>=2'), so installs are not reproducible.",
                recommendation="Pin or bound versions (for example compatible-release ranges) and consider committing a lockfile or constraints file."))
        m = re.search(r"^(config\.py):(\d+): DATA_DIR", seen, re.M | re.I) or re.search(r"^(\S+?):(\d+): .*?/home/", seen, re.M)
        if m:
            out.append(call(
                "report_violation", category="configuration", severity="medium", file=m.group(1), line=int(m.group(2)),
                description="Machine-specific absolute path in configuration",
                evidence="A data directory is hardcoded under a specific user's home directory, which will not exist on other machines or in CI.",
                recommendation="Make the path configurable via an environment variable or a path relative to the project."))
        return out


def main(argv: list[str]) -> int:
    out = Path(argv[1] if len(argv) > 1 else ROOT / "sample_report.json").resolve()
    with tempfile.TemporaryDirectory() as tmp:
        fixture = Path(tmp) / "fixture"
        build_fixture(fixture)

        def fake_fetch(ref, dest, token=None, timeout=180.0):
            shutil.copytree(fixture, dest)
            return True

        repo_access.fetch_repository = fake_fetch  # no network: use the local fixture
        return cli.main(["--repo", DEMO_URL, "--output", str(out)], llm=ScriptedLLM(StandInModel()))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
