from pathlib import Path

import pytest

from qe_agent.sandbox import NETWORK, SUT_ALIAS_URL, parse_junit, runner_cmd, write_test_dir

JUNIT_XML = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="5">
    <testcase classname="test_ts_001_budget" name="test_ts_001_neg" time="0.1">
      <failure message="assert 201 == 422">traceback here</failure>
    </testcase>
    <testcase classname="test_ts_002_ok" name="test_ts_002_happy" time="0.1"/>
    <testcase classname="test_ts_003_err" name="test_ts_003_boom" time="0.1">
      <error message="ZeroDivisionError">division by zero</error>
    </testcase>
    <testcase classname="test_ts_004_cls.TestLifecycle" name="test_ts_004_pause" time="0.1"/>
    <testcase classname="test_ts_005_param" name="test_ts_005_channels[video]" time="0.1"/>
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
    # class-based and parametrized node ids survive the round trip
    assert outcomes["test_ts_004_cls.py::TestLifecycle::test_ts_004_pause"].outcome == "passed"
    assert outcomes["test_ts_005_param.py::test_ts_005_channels[video]"].outcome == "passed"


def test_write_test_dir_blocks_path_traversal(tmp_path: Path):
    test_dir = write_test_dir(tmp_path, {"../../evil.py": "x = 1", "test_ok.py": "y = 2"})
    assert (test_dir / "evil.py").exists()  # flattened, not escaped
    assert not (tmp_path.parent / "evil.py").exists()
    assert (test_dir / "conftest.py").exists()
    assert (test_dir / "test_ok.py").read_text() == "y = 2"


def test_write_test_dir_protects_trusted_conftest(tmp_path: Path):
    from qe_agent.sandbox import CONFTEST

    test_dir = write_test_dir(tmp_path, {"conftest.py": "MALICIOUS = True"})
    assert (test_dir / "conftest.py").read_text() == CONFTEST


def test_runner_cmd_isolation_contract(tmp_path: Path):
    """The docker invocation IS the sandbox: every isolation flag must be
    present, and nothing may open a path out of the internal network."""
    tests_dir, out_dir = tmp_path / "tests", tmp_path / "out"
    cmd = runner_cmd(tests_dir, out_dir, ["."], "qe-runner-x")

    def flag(name: str) -> str:
        return cmd[cmd.index(name) + 1]

    assert flag("--network") == NETWORK
    assert "--read-only" in cmd
    assert flag("--cap-drop") == "ALL"
    assert flag("--security-opt") == "no-new-privileges"
    assert flag("--pids-limit") == "128"
    assert flag("--memory") == "512m"
    assert flag("-e") == f"SUT_BASE_URL={SUT_ALIAS_URL}"
    # test sources are read-only; the only writable host path is the out dir
    assert f"{tests_dir}:/work/tests:ro" in cmd
    assert f"{out_dir}:/work/out" in cmd
    # docker-side args only (pytest's own -p flag comes after the image name):
    # no port publishing, no host network, no privileged mode
    docker_args = cmd[: cmd.index("qe-sandbox:latest")]
    assert "-p" not in docker_args and "--publish" not in docker_args
    assert "--privileged" not in docker_args
    assert "host" not in docker_args


def test_generated_test_file_name_is_forced_to_a_safe_basename():
    """file_name is LLM output that becomes a host path in both the sandbox
    and the CLI review flow, so the schema itself normalizes it."""
    from pydantic import ValidationError

    from qe_agent.schemas import GeneratedTest

    escaped = GeneratedTest(scenario_id="TS-001", file_name="../../etc/test_evil.py", code="x = 1")
    assert escaped.file_name == "test_evil.py"
    absolute = GeneratedTest(scenario_id="TS-001", file_name="/tmp/test_x.py", code="x = 1")
    assert absolute.file_name == "test_x.py"
    for bad in ("..", "", "notpython.txt"):
        with pytest.raises(ValidationError):
            GeneratedTest(scenario_id="TS-001", file_name=bad, code="x = 1")
