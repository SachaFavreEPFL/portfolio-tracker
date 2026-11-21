"""Permet l'exécution via `python3 -m harness`."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"

if __name__ == "__main__":
    if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), "-m", "harness", *sys.argv[1:]])
    from harness.cli import main

    raise SystemExit(main())
