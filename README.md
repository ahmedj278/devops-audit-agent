# DevOps Audit Agent

An LLM-driven agent that audits a **public GitHub repository** for software-engineering and DevOps problems
(repository hygiene, dependencies, code quality, testing, CI/GitHub Actions, security, configuration,
collaboration, build/release) and writes a strict JSON report.

```bash
python agent.py --repo https://github.com/<owner>/<repo> --output report.json
```

It runs unattended, uses plain provider SDKs (Anthropic or OpenAI) and **no agent frameworks**
(no LangChain / LangGraph / n8n).

## How it works

The model is in control. It is given read-only inspection tools and three control tools, and it decides —
from what it has observed so far — which tool to call next, when to dig deeper, and when to stop.
There is no fixed checklist and no "scan then summarise" step.

```
                 +--------------------------- agent.py (CLI) ----------------------------+
                 | validate URL -> check output path -> build LLM -> fetch repo -> run loop |
                 +-----------------------------------+-----------------------------------+
                                                     |
   +--------------------------+      tool calls       v        observations
   |  LLM (Anthropic/OpenAI)  | <------------------ agent_loop.py ------------------+
   |  chooses next action     | ------------------>   |  budgets, retries,          |
   +--------------------------+                       |  context compaction,        |
                                                      |  review gate                |
                          +---------------------------+-----------------------------+
                          v                                                         v
                   tools.py (read-only)                                    findings.py (validator)
        list_files  find_files  read_file  search_code               report_violation / withdraw_violation
        git_log  inspect_workflow  scan_secrets  run_ruff            -> evidence checks, dedupe, rule IDs
                          |                                                         |
                  redaction.py  <---- all output, logs and report pass through ----+
```

| Module | Responsibility |
| --- | --- |
| `agent.py` | CLI, exit codes, logging, atomic report writing |
| `agent_loop.py` | LLM ↔ tool loop, step/time budgets, context compaction, self-review gate |
| `llm_client.py` | Anthropic + OpenAI adapters, retry/backoff, one canonical message format |
| `tools.py` | The inspection tools (all read-only) |
| `repo_access.py` | URL validation, GitHub API precheck, hardened shallow clone, tarball fallback |
| `findings.py` | Violation validation, de-duplication, rule IDs, report schema/validation |
| `redaction.py` | Secret redaction for tool output, logs and the report |
| `prompts.py` | System prompt: scope, evidence rules, false-positive guardrails, severity guide |

### Tools the model can call

| Tool | Purpose |
| --- | --- |
| `list_files(path?)` | Repository structure (big directories collapsed with counts; expandable) |
| `find_files(pattern)` | Glob lookup, used to check for a file *or its equivalent* before claiming it is missing |
| `read_file(path, start_line?, end_line?)` | Numbered, paged, redacted file contents |
| `search_code(pattern, path_glob?)` | Regex search, `path:line: text` results |
| `git_log(max_count?, path?, include_files?)` | Commit history (clone is shallow, depth 200) |
| `inspect_workflow(path)` | GitHub Actions YAML → triggers, permissions, steps, action ref kinds, test/lint/deploy commands, risk *facts* |
| `scan_secrets(path_glob?)` | Probable credentials as `file:line + kind`, with context tags — **never values** |
| `run_ruff()` | Static Python lint (`--isolated`, informational) |
| `report_violation`, `withdraw_violation`, `finish_audit` | Control tools |

`run_pytest()` is intentionally **not** implemented: running a repository's tests executes untrusted code and
this project does not ship a container/VM sandbox.

### How false positives and fabrication are limited (enforced in code, not only in the prompt)

* A finding that cites `file`/`line` is **rejected** unless the file exists and that line was actually shown to the
  model by a tool.
* "Missing X" findings carry `missing_paths`; the system checks them against the real file tree and rejects the
  finding if the path exists.
* Findings are rejected until the model has explored the repo; duplicates are rejected; rule IDs are assigned by code.
* A **review gate**: the first `finish_audit` returns the recorded findings and asks the model to re-check each for
  false positives/duplicates/inflated severity (it can `withdraw_violation`) before the audit may end.
* The prompt lists accepted equivalents (pyproject vs requirements, other lockfiles, lint via CI/pre-commit/tox…) and
  says remote settings such as branch protection cannot be judged from files.

### Safety

* The target repo is cloned into a temp dir (deleted afterwards) with hooks disabled, symlinks disabled, no
  submodules, no credential prompts, a timeout, and a size limit. Tools never write to it (covered by a test).
* Repository code is never executed. `ruff` is static analysis run with `--isolated` so repo config cannot load code.
* Paths are validated (relative only, no `..`, symlinks refused, `.git` internals hidden).
* Secrets are redacted **before** the model sees any output, in logs, and in the report (known token formats, URL
  passwords, `password=`/`token=` style assignments, multi-line private keys, and our own API key).
  Placeholders such as `${TOKEN}` or `your_key_here` are left visible so they are not mistaken for leaks.
* Subprocesses get a minimal environment — API keys and `GITHUB_TOKEN` are not passed to `git`/`ruff`.
* Repository text is treated as untrusted data in the system prompt (prompt-injection hardening).

## Installation

Requires Python 3.10+ and `git`.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then edit .env
```

`.env` (see `.env.example`):

| Variable | Meaning |
| --- | --- |
| `LLM_PROVIDER` | `anthropic` (default) or `openai` |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | key for the chosen provider (required) |
| `LLM_MODEL` | optional model override (defaults: `claude-sonnet-5-5`, `gpt-4o`) |
| `OPENAI_BASE_URL` | optional OpenAI-compatible endpoint |
| `GITHUB_TOKEN` | optional; only raises the GitHub API rate limit for the precheck |
| `AGENT_MAX_STEPS`, `AGENT_TIMEOUT_SECONDS`, `LLM_REQUEST_TIMEOUT` | optional budgets (40 turns / 900 s / 120 s) |

## Usage

```bash
python agent.py --repo https://github.com/example/demo-project --output report.json
```

Optional flags: `--max-steps N`, `--timeout SECONDS`, `--provider {anthropic,openai}`, `--model NAME`, `-v`.
Progress logs go to **stderr** (secret-redacted); stdout prints one summary line.

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Report written (also when the step/time budget ended the run — the report holds what was confirmed) |
| 2 | Invalid input: bad URL, unwritable output path |
| 3 | LLM misconfigured or unavailable after retries (a *partial* report is written only if findings exist) |
| 4 | Repository not found / private / too large / clone timeout |
| 5 | Unexpected internal error |

On fatal errors **no report is written**, so a failure is never mistaken for a clean bill of health.

### Report format

```json
{
  "repository": "https://github.com/example/demo-project",
  "violations": [
    {
      "rule_id": "SEC-001",
      "category": "security",
      "severity": "critical",
      "file": "config.py",
      "line": 1,
      "description": "Hardcoded AWS access key ID in source",
      "evidence": "…",
      "recommendation": "…"
    }
  ]
}
```

* `severity` ∈ `critical|high|medium|low`; `file`/`line` are `null` when not applicable; `violations: []` is valid.
* Categories: `repository_hygiene`, `dependency_management`, `code_quality`, `testing`, `ci_cd`, `security`,
  `configuration`, `collaboration`, `build_release`. Rule IDs are `HYG/DEP/QUA/TST/CI/SEC/CFG/COL/BLD-NNN`.
* The brief says "seven fields" but lists eight; all eight are always emitted.
* The file is written atomically after schema validation.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite uses a scripted fake LLM whose policy reads earlier tool output to choose the next call (proving that
tool selection reacts to observations), fixture git repositories built at runtime, and a fake SDK module for the
provider adapters. No network or API key is needed. Secret-shaped test values are assembled at runtime so no
credential-looking literal exists in the repository.

## About `sample_report.json`

`sample_report.json` was produced by `python scripts/generate_sample.py`, which runs the **real pipeline**
(CLI → tool loop → tools → validation → report writer) against a **local fixture repository with planted
defects**, driven by a small deterministic **stand-in policy instead of a real LLM** (no API key was available
where this project was developed). Every finding passed the same evidence validation as a real run, but the
investigation choices came from the script, and `repository` is a placeholder URL. To generate a report from a real
LLM against a real repository, run the command in *Usage* and overwrite the file.

## Limitations

* **Never run against a live LLM during development.** The Anthropic/OpenAI adapters are covered only by contract
  tests against fake SDK modules; request/response compatibility with the real SDKs, real tool-calling behaviour and
  — most importantly — **detection recall/precision on real repositories have not been measured**.
* The clean-environment `pip install -r requirements.txt` could not be verified where this was built (PyPI was not
  reachable); the test suite was run with the tools already available there. The `anthropic`/`openai` SDKs were not
  installed in that environment.
* No `run_pytest`: failing tests cannot be detected by execution, only inferred from CI configuration.
* History is a depth-200 shallow clone; commit-based findings about older history are not possible.
* Secret detection is pattern-based: it can miss unusual formats and flag test fixtures (the tool adds context tags
  and the prompt tells the model to verify). Redaction is best-effort defence in depth, not a guarantee.
* `search_code` rejects obviously catastrophic regexes but Python's `re` cannot be interrupted mid-match.
* Remote settings (branch protection, required reviews, repository secrets) are invisible to a file-based audit and
  are deliberately never reported.
* Only GitHub Actions has a dedicated parser; other CI systems are read as plain files.
* Tested on Linux only.
