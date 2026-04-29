import os
import sys

import pexpect


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

    # Accept default dataset (empty input) after prompt appears.
    child.expect(r"Select dataset .*default: qiaojin/PubMedQA\):", timeout=600)
    child.sendline("")

    # Main menu prompt
    child.expect(r"Enter your choice:", timeout=600)
    child.sendline("10")  # print ollama models

    # Back to main menu prompt
    child.expect(r"Enter your choice:", timeout=600)
    child.sendline("8")  # exit

    # Some flows ask about printing performance summary.
    try:
        child.expect(r"Would you like to print the performance summary now\? \(y/n\):", timeout=60)
        child.sendline("n")
    except pexpect.exceptions.TIMEOUT:
        pass

    child.expect(pexpect.EOF, timeout=300)
    return int(child.exitstatus or 0)


if __name__ == "__main__":
    raise SystemExit(run())

