import os
import sys

import pexpect


def _expect_send(child: pexpect.spawn, pattern: str, send: str, timeout: int = 600) -> None:
    child.expect(pattern, timeout=timeout)
    child.sendline(send)


def run() -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_path = os.path.join(repo_root, "main.py")

    child = pexpect.spawn(
        sys.executable,
        [main_path],
        cwd=repo_root,
        encoding="utf-8",
        timeout=600,
    )
    child.logfile = sys.stdout

    # Dataset selection: accept default
    _expect_send(child, r"Select dataset .*default: qiaojin/PubMedQA\):", "")

    # Main menu -> Experiments & Evaluation
    _expect_send(child, r"Enter your choice:", "7")

    # Experiments menu -> Global correlation experiment
    _expect_send(child, r"Select experiment:", "8")

    # Global correlation menu -> Pilot
    _expect_send(child, r"Select option \(1/2/3/0\) \[default=1\]:", "1")

    # After pilot completes, we return to main menu prompt. Exit.
    _expect_send(child, r"Enter your choice:", "8")

    # Performance summary print prompt (if shown) -> "n"
    # Some runs may not show it; handle both.
    try:
        _expect_send(child, r"Would you like to print the performance summary now\? \(y/n\):", "n", timeout=60)
    except pexpect.exceptions.TIMEOUT:
        pass

    child.expect(pexpect.EOF, timeout=120)
    return int(child.exitstatus or 0)


if __name__ == "__main__":
    raise SystemExit(run())

