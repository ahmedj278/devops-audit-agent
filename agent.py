#!/usr/bin/env python3
"""CLI entry point.

    python agent.py --repo https://github.com/<owner>/<repo> --output report.json

Exit codes
    0  success (report written; the model finished or a budget ran out)
    2  invalid input (bad URL / unusable output path)
    3  LLM unavailable or misconfigured (a partial report is written if any findings exist)
    4  repository could not be fetched (not found, private, too large, timeout)
    5  unexpected internal error
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path

from redaction import RedactingFilter, redact

EXIT_OK, EXIT_INPUT, EXIT_LLM, EXIT_REPO, EXIT_INTERNAL = 0, 2, 3, 4, 5

log = logging.getLogger("devops_agent")


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:
        pass


def _setup_logging(verbose: bool) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
    handler.addFilter(RedactingFilter())
    root = logging.getLogger("devops_agent")
    root.handlers[:] = [handler]
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.propagate = False


def _check_output_path(path: Path) -> None:
    """Fail fast (before cloning / paying for LLM calls) if the report cannot be written."""
    if path.exists() and path.is_dir():
        raise OSError(f"output path {path} is a directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".write-test-", dir=str(path.parent))
    os.close(fd)
    os.unlink(tmp)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agent.py", description="LLM-driven DevOps audit of a public GitHub repository.")
    p.add_argument("--repo", required=True, help="public GitHub repository URL (https://github.com/<owner>/<repo>)")
    p.add_argument("--output", required=True, help="path of the JSON report to write")
    p.add_argument("--max-steps", type=int, default=int(os.environ.get("AGENT_MAX_STEPS", "40")),
                   help="maximum LLM turns (default 40)")
    p.add_argument("--timeout", type=float, default=float(os.environ.get("AGENT_TIMEOUT_SECONDS", "900")),
                   help="wall-clock budget for the reasoning loop in seconds (default 900)")
    p.add_argument("--provider", choices=["anthropic", "openai", "gemini"], help="LLM provider (default: $LLM_PROVIDER or anthropic)")
    p.add_argument("--model", help="model name (default: $LLM_MODEL or provider default)")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p


def main(argv: list[str] | None = None, *, llm=None) -> int:
    """Run the audit. ``llm`` can be injected (tests); otherwise built from the environment."""
    _load_dotenv()
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)

    from agent_loop import AuditAgent, AuditConfig
    from findings import FindingStore, build_report, write_report
    from llm_client import ConfigError, create_llm_client
    from repo_access import InvalidInputError, RepoAccessError, fetch_repository, parse_github_url
    from tools import RepoTools

    try:
        ref = parse_github_url(args.repo)
    except InvalidInputError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_INPUT
    out_path = Path(args.output)
    try:
        _check_output_path(out_path)
    except OSError as e:
        print(f"error: cannot write output file: {e}", file=sys.stderr)
        return EXIT_INPUT
    if args.max_steps < 1:
        print("error: --max-steps must be >= 1", file=sys.stderr)
        return EXIT_INPUT

    try:
        if llm is None:
            llm = create_llm_client(args.provider, args.model)
    except ConfigError as e:
        print(f"error: LLM configuration problem: {e}", file=sys.stderr)
        return EXIT_LLM

    try:
        with tempfile.TemporaryDirectory(prefix="devops-audit-") as tmp:
            workdir = Path(tmp) / "repo"
            log.info("fetching %s", ref.url)
            try:
                fetch_repository(ref, workdir, token=os.environ.get("GITHUB_TOKEN") or None)
            except RepoAccessError as e:
                print(f"error: {redact(str(e))}", file=sys.stderr)
                return EXIT_REPO
            tools = RepoTools(workdir)
            store = FindingStore(tools)
            agent = AuditAgent(llm, tools, store, AuditConfig(max_steps=args.max_steps, max_wall_seconds=args.timeout))
            result = agent.run(ref.url)

        log.info("stopped: %s after %d step(s); tools used: %s; findings: %d; tokens in/out: %d/%d",
                 result.stop_reason, result.steps, dict(result.tool_usage), len(result.violations),
                 result.input_tokens, result.output_tokens)
        report = build_report(ref.url, result.violations)

        if result.llm_error:
            print(f"error: LLM failure: {result.llm_error}", file=sys.stderr)
            if result.violations:
                write_report(out_path, report)
                print(f"partial report with {len(result.violations)} finding(s) written to {out_path}", file=sys.stderr)
            return EXIT_LLM
        if result.stop_reason not in ("finished",):
            log.warning("audit did not finish cleanly (%s); report contains findings recorded so far", result.stop_reason)
        write_report(out_path, report)
        print(f"Report written to {out_path} ({len(result.violations)} violation(s))")
        return EXIT_OK
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as e:  # last-resort guard: always a useful message + nonzero exit
        print(f"error: unexpected failure: {type(e).__name__}: {redact(str(e))}", file=sys.stderr)
        if args.verbose:
            print(redact(traceback.format_exc()), file=sys.stderr)
        return EXIT_INTERNAL


if __name__ == "__main__":
    sys.exit(main())
