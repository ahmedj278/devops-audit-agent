"""Tests of the reasoning loop. The scripted policies *read the previous tool outputs* to pick
the next action, which demonstrates that control flow is driven by observations."""

from agent_loop import AuditAgent, AuditConfig
from findings import FindingStore
from llm_client import LLMUnavailableError
from tests.fakes import ScriptedLLM, all_results_text, call, last_results, reply
from tools import RepoTools


def make_agent(repo, policy, **cfg):
    tools = RepoTools(repo)
    store = FindingStore(tools)
    llm = ScriptedLLM(policy)
    return AuditAgent(llm, tools, store, AuditConfig(**cfg)), llm, tools, store


def tool_sequence(messages):
    return [b["name"] for m in messages if m["role"] == "assistant" for b in m["content"] if b.get("type") == "tool_use"]


def reactive_policy(messages, n):
    """Orients, then follows leads it sees in observations."""
    seq = tool_sequence(messages)
    seen = all_results_text(messages)
    if n == 1:
        return reply(call("list_files"))
    if ".github/workflows/ci.yml" in seen and "inspect_workflow" not in seq:
        return reply(call("inspect_workflow", path=".github/workflows/ci.yml"))
    if "pull_request_target" in seen and "read_file" not in seq:
        return reply(call("read_file", path=".github/workflows/ci.yml"))
    if "checks out untrusted PR head" in seen and "report_violation" not in seq:
        return reply(call("report_violation", category="security", severity="critical",
                          file=".github/workflows/ci.yml", line=2,
                          description="pull_request_target workflow checks out PR head code",
                          evidence="Workflow triggers on pull_request_target and actions/checkout uses the PR head sha",
                          recommendation="Use pull_request, or avoid checking out untrusted code in privileged context"))
    if "scan_secrets" not in seq and "report_violation" in seq and n < 8:
        return reply(call("scan_secrets"))
    if "config.py:1" in seen and seq.count("read_file") < 2:
        return reply(call("read_file", path="config.py"))
    return reply(call("finish_audit"))


def test_tool_choice_follows_observations(bad_repo, good_repo):
    agent, llm, *_ = make_agent(bad_repo, reactive_policy)
    res = agent.run("https://github.com/x/bad")
    assert res.finished and res.tool_usage["inspect_workflow"] == 1 and res.tool_usage["scan_secrets"] == 1
    assert res.violations and res.violations[0]["file"] == ".github/workflows/ci.yml"

    # same policy on a repo with a different shape takes a different path (no pull_request_target lead)
    agent2, llm2, *_ = make_agent(good_repo, reactive_policy)
    res2 = agent2.run("https://github.com/x/good")
    assert res2.tool_usage["read_file"] == 0 and res2.violations == []
    assert res.tool_usage != res2.tool_usage


def test_system_prompt_and_tools_offered(bad_repo):
    agent, llm, *_ = make_agent(bad_repo, lambda m, n: reply(call("finish_audit")))
    agent.run("u")
    assert "untrusted DATA" in llm.seen_systems[0]
    names = set(llm.tool_names[0])
    assert {"list_files", "read_file", "search_code", "git_log", "report_violation", "finish_audit"} <= names


def test_review_gate_and_withdraw(bad_repo):
    def policy(messages, n):
        if n == 1:
            return reply(call("list_files"), call("find_files", pattern="README*"), call("read_file", path="app.py"))
        if n == 2:
            return reply(call("report_violation", category="repository_hygiene", severity="low", file=None, line=None,
                              description="README is missing", evidence="find_files('README*') returned nothing",
                              recommendation="Add a README.md with setup instructions", missing_paths=["README*"]),
                         call("report_violation", category="security", severity="high", file=None, line=None,
                              description="Questionable claim", evidence="This one will be withdrawn during review",
                              recommendation="Withdraw it during review please"))
        if n == 3:
            return reply(call("finish_audit"))
        if n == 4:
            assert "Review gate" in last_results(messages)[0]
            assert "SEC-001" in last_results(messages)[0]
            return reply(call("withdraw_violation", rule_id="SEC-001", reason="not supported"), call("finish_audit"))
        raise AssertionError("loop should have ended")

    agent, llm, *_ = make_agent(bad_repo, policy)
    res = agent.run("u")
    assert res.finished and llm.n == 4
    assert [v["rule_id"] for v in res.violations] == ["HYG-001"]


def test_rejected_finding_is_fed_back_and_can_be_fixed(bad_repo):
    state = {}

    def policy(messages, n):
        if n == 1:
            return reply(call("list_files"))
        if n == 2:
            return reply(call("report_violation", category="security", severity="high", file="config.py", line=1,
                              description="Hardcoded key in config", evidence="Credential literal assigned at module level",
                              recommendation="Move the key to environment variables and rotate it"))
        if n == 3:
            state["feedback"] = last_results(messages)[0]
            return reply(call("read_file", path="config.py"))
        if n == 4:
            return reply(call("report_violation", category="security", severity="high", file="config.py", line=1,
                              description="Hardcoded key in config", evidence="Credential literal assigned at module level",
                              recommendation="Move the key to environment variables and rotate it"))
        if n == 5:
            return reply(call("finish_audit"))
        return reply(call("finish_audit"))

    agent, *_ = make_agent(bad_repo, policy, min_inspections=1)
    res = agent.run("u")
    assert state["feedback"].startswith("REJECTED") and "never shown" in state["feedback"]
    assert len(res.violations) == 1 and res.violations[0]["line"] == 1


def test_tool_errors_and_unknown_tools_do_not_crash(bad_repo):
    seen = {}

    def policy(messages, n):
        if n == 1:
            return reply(call("read_file", path="../../etc/passwd"), call("teleport"), call("read_file", nope=1))
        if n == 2:
            seen["r"] = last_results(messages)
            return reply(call("list_files"), call("find_files", pattern="*"), call("git_log"))
        return reply(call("finish_audit"))

    agent, *_ = make_agent(bad_repo, policy)
    res = agent.run("u")
    assert res.finished
    assert all(r.startswith("Error") for r in seen["r"])


def test_invalid_json_arguments_reported(bad_repo):
    def policy(messages, n):
        if n == 1:
            return reply(call("read_file", __invalid_json__="{oops"))
        if n == 2:
            assert "not valid JSON" in last_results(messages)[0]
            return reply(call("list_files"), call("find_files", pattern="*"), call("git_log"), call("finish_audit"))
        raise AssertionError("loop should have finished on turn 2")

    res = make_agent(bad_repo, policy)[0].run("u")
    assert res.finished


def test_early_finish_is_pushed_back_once(bad_repo):
    msgs = []

    def policy(messages, n):
        msgs.append(last_results(messages))
        return reply(call("finish_audit"))

    agent, llm, *_ = make_agent(bad_repo, policy)
    res = agent.run("u")
    assert llm.n == 2 and "barely inspected" in msgs[1][0] and res.finished


def test_no_tool_calls_nudged_then_stops(bad_repo):
    agent, llm, *_ = make_agent(bad_repo, lambda m, n: reply(text="I think we are done."))
    res = agent.run("u")
    assert res.stop_reason == "model_stopped_without_finish" and llm.n == 3 and not res.finished


def test_step_budget_enforced_and_findings_kept(bad_repo):
    def policy(messages, n):
        if n == 1:
            return reply(call("list_files"))
        if n == 2:
            return reply(call("report_violation", category="security", severity="high", file=None, line=None,
                              description="Something definitely wrong", evidence="Observed in the earlier tool output",
                              recommendation="Fix the observed problem promptly"))
        return reply(call("search_code", pattern="import"))

    agent, llm, *_ = make_agent(bad_repo, policy, max_steps=5)
    res = agent.run("u")
    assert res.stop_reason == "max_steps_reached" and llm.n == 5 and len(res.violations) == 1


def test_wall_clock_budget(bad_repo):
    agent, llm, *_ = make_agent(bad_repo, lambda m, n: reply(call("list_files")), max_wall_seconds=-1)
    assert agent.run("u").stop_reason == "time_budget_exhausted" and llm.n == 0


def test_llm_failure_keeps_partial_findings(bad_repo):
    def policy(messages, n):
        if n == 1:
            return reply(call("list_files"))
        if n == 2:
            return reply(call("report_violation", category="security", severity="high", file=None, line=None,
                              description="Something definitely wrong", evidence="Observed in the earlier tool output",
                              recommendation="Fix the observed problem promptly"))
        return LLMUnavailableError("down")

    res = make_agent(bad_repo, policy)[0].run("u")
    assert res.llm_error and "down" in res.llm_error and len(res.violations) == 1 and not res.finished


def test_parallel_call_cap(bad_repo):
    seen = {}

    def policy(messages, n):
        if n == 1:
            return reply(*[call("find_files", pattern="*") for _ in range(11)])
        seen["r"] = last_results(messages)
        return reply(call("finish_audit"), call("list_files"), call("find_files", pattern="x"))

    make_agent(bad_repo, policy)[0].run("u")
    assert sum(r.startswith("Skipped") for r in seen["r"]) == 3


def test_old_tool_output_is_compacted(tmp_path):
    from tests.conftest import make_repo
    repo = make_repo(tmp_path / "big", {"big.txt": "\n".join(f"line {i} " + "x" * 40 for i in range(100))})
    snapshots = []

    def policy(messages, n):
        if len(messages) > 2:  # messages[2] = first tool-result message
            snapshots.append(messages[2]["content"][0]["content"])
        if n <= 12:
            return reply(call("read_file", path="big.txt"))
        return reply(call("finish_audit"))

    agent, *_ = make_agent(repo, policy, compaction_chars=1500)
    agent.run("u")
    assert "elided" not in snapshots[0]          # fresh output is intact
    assert "older tool output elided" in snapshots[-1]  # once old and over budget, it is shortened
    assert len(snapshots[-1]) < len(snapshots[0])
