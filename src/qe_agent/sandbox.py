import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import httpx

from qe_agent import config

IMAGE = "qe-sandbox:latest"
NETWORK = "qe-sandbox-net"
SUT_CONTAINER = "qe-sut"
SUT_ALIAS_URL = "http://qe-sut:8000"

CONFTEST = """\
import os

import httpx
import pytest


@pytest.fixture()
def client():
    base_url = os.environ["SUT_BASE_URL"]
    with httpx.Client(base_url=base_url, timeout=10.0) as c:
        yield c
"""


@dataclass
class CaseOutcome:
    outcome: str  # passed | failed | error | skipped
    message: str = ""
    traceback: str = ""


def _docker(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def ensure_infra() -> None:
    """Build the sandbox image if missing and create the internal network."""
    if _docker("image", "inspect", IMAGE).returncode != 0:
        print(f"building sandbox image {IMAGE} (first run only)...")
        build = _docker(
            "build",
            "-t",
            IMAGE,
            "-f",
            str(config.PROJECT_ROOT / "docker" / "Dockerfile.sandbox"),
            str(config.PROJECT_ROOT / "docker"),
            timeout=600,
        )
        if build.returncode != 0:
            raise RuntimeError(f"sandbox image build failed:\n{build.stderr[-2000:]}")
    if _docker("network", "inspect", NETWORK).returncode != 0:
        created = _docker("network", "create", "--internal", NETWORK)
        if created.returncode != 0:
            raise RuntimeError(f"network create failed: {created.stderr}")


def start_sut_container(
    host_port: int, flaky_every: str | None = None, ready_timeout: float = 20.0
) -> None:
    """Run the SUT dual-homed: published port for the host (grounding),
    internal network for the runner. The SUT is trusted code; only the
    runner is confined to the internal-only network.

    Blocks until the SUT answers GET /health (docker run -d alone only
    proves the container started, not that uvicorn is accepting)."""
    stop_sut_container()
    args = [
        "run",
        "-d",
        "--rm",
        "--name",
        SUT_CONTAINER,
        "-p",
        f"{host_port}:8000",
        "-v",
        f"{config.PROJECT_ROOT / 'sut'}:/work/sut:ro",
        "-w",
        "/work",
    ]
    if flaky_every is not None:
        args += ["-e", f"SUT_FLAKY_EVERY={flaky_every}"]
    args += [IMAGE, "python", "-m", "sut", "--host", "0.0.0.0", "--port", "8000"]
    started = _docker(*args)
    if started.returncode != 0:
        raise RuntimeError(f"SUT container failed to start: {started.stderr}")
    connected = _docker("network", "connect", NETWORK, SUT_CONTAINER)
    if connected.returncode != 0:
        stop_sut_container()
        raise RuntimeError(f"could not attach SUT to sandbox network: {connected.stderr}")

    health_url = f"http://127.0.0.1:{host_port}/health"
    deadline = time.monotonic() + ready_timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(health_url, timeout=1.0).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.4)
    stop_sut_container()
    raise RuntimeError(f"SUT container did not become healthy at {health_url}")


def stop_sut_container() -> None:
    _docker("rm", "-f", SUT_CONTAINER)


def runner_cmd(test_dir: Path, out_dir: Path, targets: list[str], name: str) -> list[str]:
    """The full isolation contract for one pytest run, as a docker CLI call.

    Generated code gets NO writable host-backed path except out_dir, a
    per-run directory that only receives the JUnit report; the test sources
    themselves are mounted read-only so a test cannot tamper with its own
    inputs. (Residual risk — filling the disk through out_dir — is accepted
    for this exercise and noted in the design doc.)"""
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        NETWORK,
        "--read-only",
        "--tmpfs",
        "/tmp",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "128",
        "--memory",
        "512m",
        "--cpus",
        "1",
        "-v",
        f"{test_dir}:/work/tests:ro",
        "-v",
        f"{out_dir}:/work/out",
        "-w",
        "/work/tests",
        "-e",
        f"SUT_BASE_URL={SUT_ALIAS_URL}",
        IMAGE,
        "python",
        "-m",
        "pytest",
        *targets,
        "-q",
        "--tb=short",
        "-p",
        "no:cacheprovider",
        "--junitxml",
        "/work/out/junit.xml",
        "-o",
        "addopts=",  # neutralize the project's addopts (-m 'not integration')
    ]


class DockerRunner:
    """Runs generated pytest modules in the confined runner container."""

    name = "docker"

    def __init__(self, timeout: int = 300):
        self.timeout = timeout

    def run(self, test_dir: Path, node_ids: list[str] | None = None) -> dict[str, CaseOutcome]:
        """Run all tests (or only node_ids) and return outcome per pytest nodeid."""
        out_dir = test_dir.parent / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        junit_path = out_dir / "junit.xml"
        junit_path.unlink(missing_ok=True)
        container = f"qe-runner-{uuid.uuid4().hex[:8]}"
        cmd = runner_cmd(test_dir, out_dir, node_ids if node_ids else ["."], container)
        try:
            subprocess.run(cmd, capture_output=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            _docker("rm", "-f", container)
            return {
                nid: CaseOutcome(outcome="error", message="sandbox timeout")
                for nid in (node_ids or [])
            }
        if not junit_path.exists():
            return {}
        return parse_junit(junit_path)


def parse_junit(path: Path) -> dict[str, CaseOutcome]:
    """Map pytest nodeids to outcomes from a JUnit XML report."""
    outcomes: dict[str, CaseOutcome] = {}
    root = ET.parse(path).getroot()
    for case in root.iter("testcase"):
        classname = case.get("classname", "")
        name = case.get("name", "")
        # pytest emits classname as "module" or "module.TestClass" (generated
        # tests live flat in one directory, so there are no package segments).
        parts = classname.split(".") if classname else []
        if len(parts) >= 2 and parts[-1][:1].isupper():
            node_id = f"{parts[-2]}.py::{parts[-1]}::{name}"
        elif parts:
            node_id = f"{parts[-1]}.py::{name}"
        else:
            node_id = name

        failure = case.find("failure")
        error = case.find("error")
        skipped = case.find("skipped")
        if failure is not None:
            outcomes[node_id] = CaseOutcome(
                outcome="failed",
                message=(failure.get("message") or "")[:500],
                traceback=(failure.text or "")[:2000],
            )
        elif error is not None:
            outcomes[node_id] = CaseOutcome(
                outcome="error",
                message=(error.get("message") or "")[:500],
                traceback=(error.text or "")[:2000],
            )
        elif skipped is not None:
            outcomes[node_id] = CaseOutcome(outcome="skipped")
        else:
            outcomes[node_id] = CaseOutcome(outcome="passed")
    return outcomes


def write_test_dir(run_dir: Path, files: dict[str, str]) -> Path:
    """Materialize generated test files plus our own trusted conftest."""
    test_dir = run_dir / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "conftest.py").write_text(CONFTEST)
    for file_name, code in files.items():
        safe_name = Path(file_name).name  # no path traversal from LLM output
        if safe_name == "conftest.py":
            continue  # reserved for the trusted fixture module above
        (test_dir / safe_name).write_text(code)
    return test_dir
