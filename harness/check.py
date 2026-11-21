#!/usr/bin/env python3
"""Boucle de vérification déterministe du harness (lint, typage statique, tests)."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"

# Relance dans le .venv du projet si exécuté avec le Python système.
if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__, *sys.argv[1:]])


def _run_step(label: str, cmd: list[str], module_name: str) -> bool:
    if importlib.util.find_spec(module_name) is None:
        print(
            f"[MANQUANT] {label} : module '{module_name}' absent du .venv "
            "(installez les dépendances dev : pip install -e '.[dev]').",
            file=sys.stderr,
        )
        return False

    print(f"\n=== {label} ===")
    completed = subprocess.run(cmd, cwd=str(ROOT))
    return completed.returncode == 0


def main() -> int:
    steps = [
        (
            "Lint (ruff)",
            [sys.executable, "-m", "ruff", "check", "src", "tests", "harness", "scripts"],
            "ruff",
        ),
        (
            "Typage (mypy)",
            [sys.executable, "-m", "mypy", "src", "tests", "harness", "scripts"],
            "mypy",
        ),
        ("Tests (pytest)", [sys.executable, "-m", "pytest", "-v"], "pytest"),
    ]

    all_ok = True
    for label, cmd, module_name in steps:
        if not _run_step(label, cmd, module_name):
            all_ok = False

    print("\n=== Résultat global :", "OK ===" if all_ok else "ÉCHEC ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
