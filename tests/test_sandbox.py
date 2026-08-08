from pathlib import Path

from qe_agent.sandbox import NETWORK, SUT_ALIAS_URL, parse_junit, runner_cmd, write_test_dir

JUNIT_XML = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="3">
    <testcase classname="test_ts_001_budget" name="test_ts_001_neg" time="0.1">
      <failure message="assert 201 == 422">traceback here</failure>
    </testcase>
    <testcase classname="test_ts_002_ok" name="test_ts_002_happy" time="0.1"/>
    <testcase classname="test_ts_003_err" name="test_ts_003_boom" time="0.1">
      <error message="ZeroDivisionError">division by zero</error>
    </testcase>
  </testsuite>
</testsuites>
"""


def test_parse_junit(tmp_path: Path):
    xml = tmp_path / "junit.xml"
    xml.write_text(JUNIT_XML)
    outcomes = parse_junit(xml)
    assert outcomes["test_ts_001_budget.py::test_ts_001_neg"].outcome == "failed"
    assert outcomes["test_ts_002_ok.py::test_ts_002_happy"].outcome == "passed"
    assert outcomes["test_ts_003_err.py::test_ts_003_boom"].outcome == "error"
    assert "assert 201 == 422" in outcomes["test_ts_001_budget.py::test_ts_001_neg"].message


def test_write_test_dir_blocks_path_traversal(tmp_path: Path):
    test_dir = write_test_dir(tmp_path, {"../../evil.py": "x = 1", "test_ok.py": "y = 2"})
    assert (test_dir / "evil.py").exists()  # flattened, not escaped
    assert not (tmp_path.parent / "evil.py").exists()
    assert (test_dir / "conftest.py").exists()
    assert (test_dir / "test_ok.py").read_text() == "y = 2"


def test_runner_cmd_isolation_contract(tmp_path: Path):
    """The docker invocation IS the sandbox: every isolation flag must be
    present, and nothing may open a path out of the internal network."""
    cmd = runner_cmd(tmp_path, ["."], "qe-runner-x")

    def flag(name: str) -> str:
        return cmd[cmd.index(name) + 1]

    assert flag("--network") == NETWORK
    assert "--read-only" in cmd
    assert flag("--cap-drop") == "ALL"
    assert flag("--security-opt") == "no-new-privileges"
    assert flag("--pids-limit") == "128"
    assert flag("--memory") == "512m"
    assert flag("-e") == f"SUT_BASE_URL={SUT_ALIAS_URL}"
    assert flag("-v") == f"{tmp_path}:/work/tests"
    # docker-side args only (pytest's own -p flag comes after the image name):
    # no port publishing, no host network, no privileged mode
    docker_args = cmd[: cmd.index("qe-sandbox:latest")]
    assert "-p" not in docker_args and "--publish" not in docker_args
    assert "--privileged" not in docker_args
    assert "host" not in docker_args
