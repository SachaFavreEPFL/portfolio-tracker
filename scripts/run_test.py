#!/usr/bin/env python3
"""Lance et affiche les résultats de tous les tests unitaires du projet."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"

# Relance les tests dans le .venv du projet si exécuté avec le Python système.
if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__, *sys.argv[1:]])

import pytest  # noqa: E402


def main() -> int:
    return pytest.main(["-v", str(ROOT / "tests"), str(ROOT / "harness" / "tests")])


if __name__ == "__main__":
    raise SystemExit(main())
