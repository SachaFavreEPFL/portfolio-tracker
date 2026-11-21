#!/usr/bin/env python3
"""Point d'entrée CLI du harness d'agents (Ollama local & Gemini API)."""

import argparse
from pathlib import Path
import sys
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.providers import DEFAULT_OLLAMA_URL, LLMProvider, create_provider  # noqa: E402
from harness.roles import ROLES, get_role  # noqa: E402
from harness.runner import run_agent  # noqa: E402


def _parse_role_models(
    entries: list[str],
    *,
    ollama_url: str,
    think: bool,
) -> dict[str, LLMProvider]:
    """Parse les options --role-model ROLE=MODELE en fournisseurs dédiés."""
    providers: dict[str, LLMProvider] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Format invalide pour --role-model : '{entry}' (attendu ROLE=MODELE).")
        role_name, model_name = entry.split("=", 1)
        role_name = role_name.strip()
        model_name = model_name.strip()
        get_role(role_name)
        if not model_name:
            raise ValueError(f"Nom de modèle manquant pour le rôle '{role_name}'.")
        providers[role_name] = create_provider(
            model_name,
            ollama_url=ollama_url,
            think=think,
        )
    return providers


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Harness d'agents modulaires (Ollama local ou Gemini API)."
    )
    parser.add_argument(
        "--agent",
        "-a",
        choices=sorted(ROLES.keys()),
        required=True,
        help="Rôle d'agent à exécuter.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Instruction à exécuter (si omis, ouvre le mode interactif).",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Modèle à utiliser (ex. qwen3.8, gemini-4-hydrogen, gemini-4-neon).",
    )
    parser.add_argument(
        "--role-model",
        action="append",
        default=[],
        metavar="ROLE=MODELE",
        help="Modèle spécifique pour un sous-agent (répétable, ex. --role-model coder=gemini-4-hydrogen).",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_OLLAMA_URL,
        help="URL du serveur Ollama local.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Nombre maximal d'itérations d'outils (défaut : 15).",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Désactive le mode thinking sur Ollama.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Masque le détail des appels d'outils sur stderr.",
    )
    args = parser.parse_args()

    try:
        provider = create_provider(
            args.model,
            ollama_url=args.url,
            think=not args.no_think,
        )
        role_providers = _parse_role_models(
            args.role_model,
            ollama_url=args.url,
            think=not args.no_think,
        )
        print(f"Rôle : {args.agent} | Modèle : {provider.model}", file=sys.stderr)
        if role_providers:
            overrides_str = ", ".join(f"{r}={p.model}" for r, p in role_providers.items())
            print(f"Modèles par sous-agent : {overrides_str}", file=sys.stderr)

        if args.prompt:
            result = run_agent(
                args.prompt,
                role_name=args.agent,
                provider=provider,
                root=ROOT,
                max_steps=args.max_steps,
                verbose=not args.quiet,
                role_providers=role_providers,
            )
            print(result.output)
            return 0 if result.status == "completed" else 1

        print(f"Mode interactif [{args.agent}] (tapez 'exit' ou Ctrl+C pour quitter).")
        while True:
            try:
                user_input = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not user_input or user_input.lower() in {"exit", "quit"}:
                return 0
            result = run_agent(
                user_input,
                role_name=args.agent,
                provider=provider,
                root=ROOT,
                max_steps=args.max_steps,
                verbose=not args.quiet,
                role_providers=role_providers,
            )
            print(result.output)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"Erreur HTTP {exc.code} : {body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Impossible de joindre le serveur : {exc.reason}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
