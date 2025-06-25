import subprocess
import sys


def test_success(tmp_path):
    doc1 = tmp_path / "a.txt"
    doc1.write_text("alpha beta gamma")
    doc2 = tmp_path / "b.txt"
    doc2.write_text("gamma delta epsilon")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "finchat_sec_qa.cli",
            "gamma",
            str(doc1),
            str(doc2),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "alpha beta gamma" in result.stdout
    assert "b.txt" in result.stdout


def test_edge_case_invalid_input():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "finchat_sec_qa.cli",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
