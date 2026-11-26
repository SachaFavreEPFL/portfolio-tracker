# AGENTS.md — Portfolio Tracker

## Objectif du projet
Outil Python CLI mono-utilisateur de suivi de portefeuille d'actions, développé de façon incrémentale.

## Architecture
- `src/portfolio_tracker/` : code métier pur (aucune dépendance vers `harness/` ou `scripts/`).
  - `contracts.py` : modèles Pydantic immuables (`ConfigDict(frozen=True)`) et `Protocol` d'interface.
  - `market_data.py` : fournisseur `yfinance` avec cache JSON local (TTL 24h, écriture atomique).
- `tests/` : tests unitaires isolés (`pytest`, mocks, `tmp_path`, aucun appel réseau réel).
- `scripts/` : lanceurs pratiques (`run.py` pour l'exécution réelle, `run_test.py` pour `pytest`).
- `harness/` : boucle de vérification déterministe (`check.py` : `ruff`, `mypy`, `pytest`).
- `.agents/skills/` : rôles d'agents utilisables directement depuis le chat (`coder`, `test`, `doc`, `audit`, `verifier`, `orchestrator`).

## Mode `/swarm` (multi-agents parallèle)
- **Quand l'utiliser** : uniquement pour les tâches qui se découpent en couloirs (*lanes*) indépendants (plusieurs modules, code + tests + doc). Pour une correction simple sur 1 fichier, préférer le mode direct.
- **Coordinator** : la conversation principale planifie et répartit les *lanes* sans éditer le code directement.
- **Workers (`coder`, `test`, `doc`)** : chaque worker suit sa skill `.agents/skills/<role>/SKILL.md` et ne modifie que les fichiers de sa *lane* (utiliser `workspace: "share"` si des éditions risquent de se chevaucher).
- **Vérification indépendante (`verifier`)** : un worker `@verifier` (`.agents/skills/verifier/SKILL.md`) lit les diffs et exécute `python3 harness/check.py` avant de rendre `PASS` ou `REJECT`.

## Conventions
- Python `>=3.11`, annotations de types complètes (`str | None`, `list[str]`, etc.).
- Séparer les contrats (`Protocol` / modèles Pydantic) des implémentations concrètes.
- Commentaires en français uniquement pour expliquer le *pourquoi* (pas de paraphrases inutiles).
- Ne jamais commiter de clés API ni de fichiers générés (`cache/`, `.venv/`, `*.egg-info/`).

## Vérification obligatoire
Avant de conclure une modification, exécuter :
```zsh
python3 harness/check.py
```
(ou `python3 scripts/run_test.py` pour lancer uniquement les tests unitaires).
