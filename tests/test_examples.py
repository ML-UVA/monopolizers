import pytest
import subprocess
import sys
import os
from pathlib import Path

def test_run_smoke_games():
    # Run the script and check it doesn't crash
    script_path = Path(__file__).parent.parent / 'examples' / 'run_smoke_games.py'
    repo_root = Path(__file__).parent.parent.parent
    
    env = os.environ.copy()
    env['PYTHONPATH'] = str(repo_root)
    
    result = subprocess.run(
        [sys.executable, str(script_path)], 
        capture_output=True, 
        text=True, 
        timeout=30,
        env=env
    )
    assert result.returncode == 0, f"Script failed with output:\n{result.stderr}"

def test_train_vs_greedy():
    # Similar smoke test
    script_path = Path(__file__).parent.parent / 'examples' / 'train_vs_greedy.py'
    repo_root = Path(__file__).parent.parent.parent
    
    env = os.environ.copy()
    env['PYTHONPATH'] = str(repo_root)
    
    result = subprocess.run(
        [sys.executable, str(script_path)], 
        capture_output=True, 
        text=True, 
        timeout=30,
        env=env
    )
    assert result.returncode == 0, f"Script failed with output:\n{result.stderr}"
