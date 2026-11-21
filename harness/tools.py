"""Outils sécurisés exposés aux agents du harness avec garde-fous par rôle."""

import http.client
import os
import subprocess
import sys
import urllib.error
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.providers import ProviderError

IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "cache",
    "node_modules",
    "dist",
    "build",
}
SENSITIVE_FILENAMES = {
    ".env",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
}
SENSITIVE_SUFFIXES = (
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".jks",
)
MAX_OUTPUT_CHARS = 24_000


def _is_ignored_dir(name: str) -> bool:
    return name in IGNORED_DIRS or name.endswith(".egg-info")


def _is_sensitive_file(name: str) -> bool:
    lower = name.lower()
    return (
        lower in SENSITIVE_FILENAMES
        or lower.startswith(".env.")
        or lower.endswith(SENSITIVE_SUFFIXES)
    )


TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "list_files": {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Liste les fichiers du projet dans un dossier donné.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Chemin relatif du dossier (par défaut '.').",
                    }
                },
            },
        },
    },
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Lit le contenu d'un fichier du projet (avec numéros de lignes optionnels).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Chemin relatif du fichier à lire.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Ligne de début (1-indexée, optionnelle).",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Ligne de fin incluse (1-indexée, optionnelle).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    "edit_file": {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Modifie une portion précise d'un fichier existant. "
                "Échoue si 'target' est introuvable ou ambigu (plus d'une occurrence)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Chemin relatif du fichier à modifier.",
                    },
                    "target": {
                        "type": "string",
                        "description": "Extrait exact (unique dans le fichier) à remplacer.",
                    },
                    "replacement": {
                        "type": "string",
                        "description": "Nouveau texte de remplacement.",
                    },
                },
                "required": ["path", "target", "replacement"],
            },
        },
    },
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Crée un nouveau fichier ou remplace entièrement un fichier existant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Chemin relatif du fichier à écrire.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Contenu complet du fichier.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    "run_checks": {
        "type": "function",
        "function": {
            "name": "run_checks",
            "description": "Exécute la suite de vérifications déterministes du projet (harness/check.py).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    "git_status": {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Affiche l'état Git court du dépôt (fichiers modifiés ou non suivis).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    "git_diff": {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Affiche le diff Git des modifications non commitées.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Chemin relatif optionnel pour filtrer le diff.",
                    }
                },
            },
        },
    },
    "delegate_to_agent": {
        "type": "function",
        "function": {
            "name": "delegate_to_agent",
            "description": "Délègue une sous-tâche précise à un agent spécialisé ('coder', 'test', 'doc' ou 'audit').",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "enum": ["coder", "test", "doc", "audit"],
                        "description": "Nom du rôle cible.",
                    },
                    "task": {
                        "type": "string",
                        "description": "Instruction claire et complète pour l'agent cible.",
                    },
                },
                "required": ["agent_name", "task"],
            },
        },
    },
}


def resolve_safe_path(root: Path, relative_path: str) -> Path:
    """Résout un chemin relatif en vérifiant qu'il reste dans le projet et hors zones sensibles."""
    root_resolved = root.resolve()
    target = (root_resolved / relative_path).resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise ValueError(f"Accès refusé hors du projet : {relative_path}")
    if target != root_resolved:
        rel_parts = target.relative_to(root_resolved).parts
        if any(_is_ignored_dir(part) for part in rel_parts):
            raise ValueError(f"Accès refusé à un dossier protégé : {relative_path}")
        if any(_is_sensitive_file(part) for part in rel_parts):
            raise ValueError(f"Accès refusé à un fichier sensible : {relative_path}")
    return target


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [tronqué : {len(text) - limit} caractères restants]"


def _require_str(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Paramètre '{key}' invalide (chaîne de caractères attendue).")
    return value


@dataclass(frozen=True)
class ToolExecutionResult:
    """Résultat structuré d'un appel d'outil, indiquant si le workspace a été modifié."""

    output: str
    mutated: bool = False


class ToolBox:
    """Boîte à outils bornée pour un agent donné (permissions et préfixes d'écriture)."""

    def __init__(
        self,
        root: Path,
        allowed_tools: set[str],
        *,
        allowed_write_prefixes: tuple[str, ...] | None = None,
        delegate_fn: Callable[[str, str], str | ToolExecutionResult] | None = None,
    ) -> None:
        self._root = root.resolve()
        self._allowed_tools = set(allowed_tools)
        self._allowed_write_prefixes = allowed_write_prefixes
        self._delegate_fn = delegate_fn

    def schemas(self) -> list[dict[str, Any]]:
        """Retourne uniquement les schémas des outils autorisés pour cet agent."""
        return [
            TOOL_SCHEMAS[name]
            for name in TOOL_SCHEMAS
            if name in self._allowed_tools
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Exécute un outil autorisé et renvoie sa sortie texte sans lever d'exception."""
        return self.execute_with_status(name, arguments).output

    def execute_with_status(self, name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
        """Exécute un outil autorisé et renvoie sa sortie ainsi qu'un indicateur de mutation."""
        if name not in self._allowed_tools:
            return ToolExecutionResult(f"Erreur : outil non autorisé pour cet agent : {name}")
        try:
            if name == "list_files":
                directory = arguments.get("directory")
                if directory is None or directory == "":
                    return ToolExecutionResult(self._list_files("."))
                if not isinstance(directory, str):
                    raise ValueError(
                        "Paramètre 'directory' invalide (chaîne de caractères attendue)."
                    )
                return ToolExecutionResult(self._list_files(directory))
            if name == "read_file":
                return ToolExecutionResult(
                    self._read_file(
                        _require_str(arguments, "path"),
                        start_line=arguments.get("start_line"),
                        end_line=arguments.get("end_line"),
                    )
                )
            if name == "edit_file":
                return self._edit_file(
                    _require_str(arguments, "path"),
                    _require_str(arguments, "target"),
                    _require_str(arguments, "replacement"),
                )
            if name == "write_file":
                return self._write_file(
                    _require_str(arguments, "path"),
                    _require_str(arguments, "content"),
                )
            if name == "run_checks":
                return ToolExecutionResult(self._run_checks())
            if name == "git_status":
                return ToolExecutionResult(self._run_git(["status", "--short"]))
            if name == "git_diff":
                rel_path = arguments.get("path")
                if rel_path is not None and rel_path != "":
                    if not isinstance(rel_path, str):
                        raise ValueError(
                            "Paramètre 'path' invalide (chaîne de caractères attendue)."
                        )
                    resolved = resolve_safe_path(self._root, rel_path)
                    return ToolExecutionResult(
                        self._run_git(["diff", "--", str(resolved.relative_to(self._root))])
                    )
                return ToolExecutionResult(self._run_git(["diff"]))
            if name == "delegate_to_agent":
                if self._delegate_fn is None:
                    return ToolExecutionResult("Erreur : délégation non configurée.")
                delegated = self._delegate_fn(
                    _require_str(arguments, "agent_name"),
                    _require_str(arguments, "task"),
                )
                if isinstance(delegated, ToolExecutionResult):
                    return delegated
                return ToolExecutionResult(delegated)
            return ToolExecutionResult(f"Outil inconnu : {name}")
        except (ProviderError, urllib.error.URLError, TimeoutError, http.client.HTTPException):
            raise
        except Exception as exc:
            return ToolExecutionResult(f"Erreur lors de l'exécution de {name} : {exc}")

    def _check_write_allowed(self, target: Path) -> None:
        if self._allowed_write_prefixes is None:
            return
        rel = target.relative_to(self._root).as_posix()
        if not any(rel.startswith(prefix) for prefix in self._allowed_write_prefixes):
            allowed = ", ".join(self._allowed_write_prefixes)
            raise PermissionError(
                f"Écriture refusée sur '{rel}' (chemins autorisés : {allowed})."
            )

    def _list_files(self, directory: str = ".") -> str:
        base = resolve_safe_path(self._root, directory)
        if not base.exists() or not base.is_dir():
            return f"Dossier introuvable : {directory}"

        files: list[str] = []
        for current, dirs, filenames in os.walk(base):
            dirs[:] = sorted(d for d in dirs if not _is_ignored_dir(d))
            for filename in sorted(filenames):
                if _is_sensitive_file(filename):
                    continue
                rel = (Path(current) / filename).relative_to(self._root)
                files.append(rel.as_posix())
        return "\n".join(files) if files else "(aucun fichier)"

    def _read_file(
        self,
        path: str,
        *,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        target = resolve_safe_path(self._root, path)
        if not target.is_file():
            return f"Fichier introuvable : {path}"

        lines = target.read_text(encoding="utf-8").splitlines()
        start = max(1, int(start_line)) if start_line is not None else 1
        end = min(len(lines), int(end_line)) if end_line is not None else len(lines)
        numbered = [f"{idx}: {lines[idx - 1]}" for idx in range(start, end + 1)]
        return _truncate("\n".join(numbered)) if numbered else "(fichier vide)"

    def _edit_file(self, path: str, target_text: str, replacement: str) -> ToolExecutionResult:
        target = resolve_safe_path(self._root, path)
        self._check_write_allowed(target)
        if not target.is_file():
            return ToolExecutionResult(f"Fichier introuvable : {path}")
        if not target_text:
            return ToolExecutionResult("Erreur : 'target' ne peut pas être vide.")

        content = target.read_text(encoding="utf-8")
        occurrences = content.count(target_text)
        if occurrences == 0:
            return ToolExecutionResult(f"Erreur : texte cible introuvable dans {path}.")
        if occurrences > 1:
            return ToolExecutionResult(
                f"Erreur : texte cible ambigu ({occurrences} occurrences dans {path}). "
                "Fournissez davantage de contexte."
            )

        updated = content.replace(target_text, replacement, 1)
        target.write_text(updated, encoding="utf-8")
        return ToolExecutionResult(f"Fichier modifié avec succès : {path}", mutated=True)

    def _write_file(self, path: str, content: str) -> ToolExecutionResult:
        target = resolve_safe_path(self._root, path)
        self._check_write_allowed(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolExecutionResult(f"Fichier écrit avec succès : {path}", mutated=True)

    def _run_checks(self) -> str:
        check_script = self._root / "harness" / "check.py"
        if not check_script.is_file():
            return "Erreur : harness/check.py introuvable."
        completed = subprocess.run(
            [sys.executable, str(check_script)],
            cwd=str(self._root),
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = (completed.stdout + completed.stderr).strip()
        summary = f"code={completed.returncode}\n{output}" if output else f"code={completed.returncode}"
        return _truncate(summary)

    def _run_git(self, args: list[str]) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(self._root),
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = (completed.stdout + completed.stderr).strip()
        if completed.returncode != 0:
            return f"git erreur (code={completed.returncode}) : {output}"
        return _truncate(output) if output else "(aucun changement)"
