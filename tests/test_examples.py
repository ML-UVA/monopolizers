import pytest
import subprocess
import sys

def test_run_smoke_games():
    # Run the script and check it doesn't crash
    result = subprocess.run([sys.executable, 'examples/run_smoke_games.py'], capture_output=True, text=True)
    assert result.returncode == 0  # Assuming it exits cleanly

def test_train_vs_greedy():
    # Similar smoke test
    result = subprocess.run([sys.executable, 'examples/train_vs_greedy.py'], capture_output=True, text=True, timeout=10)  # Short timeout
    assert result.returncode == 0
