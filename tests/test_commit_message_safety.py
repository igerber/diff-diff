"""Regression: /submit-pr and /push-pr-update must commit via `git commit --file`,
not a `git commit -m "$(cat <<'EOF' … )"` heredoc.

A quoted heredoc stops expansion *inside* the body, but a message whose body contains
a line equal to the delimiter (`EOF`) closes the heredoc early — the following lines
then run as shell. This test proves the mandated pattern (write the message to a file
with a non-shell writer, then `git commit --file`) is immune to that, using a message
that would `touch` a sentinel if any line were interpreted by a shell.
"""

import subprocess

import pytest


def _git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)


@pytest.fixture()
def repo(tmp_path):
    _git("init", cwd=tmp_path)
    _git("config", "user.email", "t@t.t", cwd=tmp_path)
    _git("config", "user.name", "t", cwd=tmp_path)
    (tmp_path / "f.txt").write_text("x")
    _git("add", "-A", cwd=tmp_path)
    return tmp_path


def test_commit_file_does_not_execute_message_content(repo, tmp_path):
    sentinel = tmp_path / "sentinel"
    # A message that breaks a `<<'EOF'` heredoc and, if any line were run as shell,
    # would create the sentinel.
    malicious = f"Fix thing\n\nEOF\n$(touch {sentinel})\n`touch {sentinel}`\nend\n"
    msg_file = repo / "commit-msg.txt"
    msg_file.write_text(malicious)  # non-shell write, mirrors the Write tool

    r = _git("commit", "--file", str(msg_file), cwd=repo)
    assert r.returncode == 0, r.stderr
    assert not sentinel.exists(), "commit message content executed as shell"

    # And the message is preserved verbatim (nothing stripped or interpreted).
    body = _git("log", "-1", "--pretty=%B", cwd=repo).stdout
    assert "$(touch" in body and "EOF" in body
