#!/usr/bin/env python3
"""Lance et affiche les résultats de tous les tests unitaires du projet."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def _reexec_in_venv_if_needed() -> None:
    """Relance les tests dans le .venv du projet si exécuté avec un autre interpréteur."""
    if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__, *sys.argv[1:]])


def main(argv: list[str] | None = None) -> int:
    import pytest

    extra_args = list(argv) if argv is not None else sys.argv[1:]
    return pytest.main(["-v", str(ROOT / "tests"), str(ROOT / "harness" / "tests"), *extra_args])


if __name__ == "__main__":
    _reexec_in_venv_if_needed()
    raise SystemExit(main())
