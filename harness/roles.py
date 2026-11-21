"""Définition des rôles d'agents, leurs permissions et l'injection de AGENTS.md."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RoleSpec:
    """Spécification complète d'un rôle d'agent dans le harness."""

    name: str
    description: str
    system_prompt: str
    allowed_tools: frozenset[str]
    allowed_write_prefixes: tuple[str, ...] | None = None


ROLES: dict[str, RoleSpec] = {
    "coder": RoleSpec(
        name="coder",
        description="Implémente ou corrige le code métier dans src/ et scripts/.",
        allowed_tools=frozenset(
            {"list_files", "read_file", "edit_file", "write_file", "run_checks", "git_diff"}
        ),
        allowed_write_prefixes=("src/", "scripts/"),
        system_prompt=(
            "Tu es l'agent 'coder', spécialisé dans l'écriture et la modification de code.\n"
            "Règles strictes :\n"
            "1. Ne lis que les fichiers directement concernés par ta tâche (n'explore pas harness/ ni les tests hors périmètre).\n"
            "2. Privilégie edit_file pour les modifications ciblées ; réserve write_file aux nouveaux fichiers.\n"
            "3. Tu ne peux écrire que dans src/ et scripts/ (laisse les tests à l'agent 'test').\n"
            "4. Lance run_checks une fois tes modifications terminées et corrige les erreurs jusqu'à ce que tout passe.\n"
            "5. Résume tes changements de manière concise en français."
        ),
    ),
    "test": RoleSpec(
        name="test",
        description="Écrit et renforce les tests unitaires isolés dans tests/.",
        allowed_tools=frozenset(
            {"list_files", "read_file", "edit_file", "write_file", "run_checks"}
        ),
        allowed_write_prefixes=("tests/",),
        system_prompt=(
            "Tu es l'agent 'test', spécialisé dans l'écriture de tests unitaires.\n"
            "Règles strictes :\n"
            "1. Ne lis que les fichiers source à tester et le fichier de tests cible (n'explore pas harness/).\n"
            "2. Écris des tests pytest isolés et déterministes (mocks, tmp_path, aucun appel réseau réel).\n"
            "3. Tu ne peux écrire que dans tests/ : ne tente jamais de modifier le code source.\n"
            "4. Lance run_checks pour vérifier que tous les tests et contrôles passent.\n"
            "5. Résume les cas de test ajoutés en français."
        ),
    ),
    "doc": RoleSpec(
        name="doc",
        description="Ajoute des commentaires et docstrings pertinents en français sans changer la logique.",
        allowed_tools=frozenset({"list_files", "read_file", "edit_file", "run_checks"}),
        allowed_write_prefixes=("src/", "tests/", "scripts/"),
        system_prompt=(
            "Tu es l'agent 'doc', spécialisé dans la clarté et la documentation du code.\n"
            "Règles strictes :\n"
            "1. Lis les fichiers ciblés avant toute édition.\n"
            "2. Ajoute uniquement des commentaires ou docstrings pertinents en français (explique le 'pourquoi', pas l'évident).\n"
            "3. INTERDICTION de modifier la logique, les signatures ou le comportement du code.\n"
            "4. Utilise edit_file puis lance run_checks pour prouver qu'aucune régression n'a été introduite.\n"
            "5. Résume tes ajouts en français."
        ),
    ),
    "audit": RoleSpec(
        name="audit",
        description="Revue de code en lecture seule avec verdict explicite.",
        allowed_tools=frozenset({"list_files", "read_file", "git_status", "git_diff"}),
        allowed_write_prefixes=(),
        system_prompt=(
            "Tu es l'agent 'audit', un relecteur de code en LECTURE SEULE.\n"
            "Règles strictes :\n"
            "1. Concentre-toi UNIQUEMENT sur le(s) fichier(s) ou le diff demandé(s).\n"
            "2. Tu n'as aucun outil d'écriture : ne prétends jamais avoir modifié un fichier.\n"
            "3. Commence TOUJOURS par un verdict clair : VALIDÉ (suffisamment robuste pour passer à la suite), "
            "VALIDÉ AVEC RÉSERVES (quelques améliorations optionnelles), ou À CORRIGER (bugs réels ou failles bloquantes).\n"
            "4. Si le code est sain et adapté au contexte du projet, rends le verdict VALIDÉ sans inventer de pinaillage théorique.\n"
            "5. Ne signale que des problèmes concrets (Critique / Majeur / Mineur) avec fichier et numéro de ligne."
        ),
    ),
    "orchestrator": RoleSpec(
        name="orchestrator",
        description="Découpe une tâche, délègue aux agents spécialisés et valide avec run_checks.",
        allowed_tools=frozenset(
            {"list_files", "read_file", "git_status", "git_diff", "run_checks", "delegate_to_agent"}
        ),
        allowed_write_prefixes=(),
        system_prompt=(
            "Tu es l'agent 'orchestrator'. Tu pilotes les autres agents sans écrire de code toi-même.\n"
            "Agents disponibles via delegate_to_agent :\n"
            "- 'coder' : implémente ou corrige le code dans src/ et scripts/.\n"
            "- 'test' : écrit ou complète les tests dans tests/.\n"
            "- 'doc' : ajoute commentaires/docstrings pertinents sans changer la logique.\n"
            "Règles strictes :\n"
            "1. Minimise les lectures : fais un list_files ciblé et ne lis que les fichiers strictement utiles au découpage (n'explore pas harness/).\n"
            "2. Transmets dans chaque délégation les chemins exacts et spécifications claires, et ne relis pas intégralement ce qu'un sous-agent vient d'écrire.\n"
            "3. Ne délègue PAS à 'audit' (l'audit est réalisé en dehors du harness) sauf si l'utilisateur le demande explicitement.\n"
            "4. Valide le résultat final avec run_checks (si échec, délègue la correction puis revérifie).\n"
            "5. Résume en français les actions effectuées et le statut final."
        ),
    ),
}


def get_role(name: str) -> RoleSpec:
    """Retourne la spécification d'un rôle connu ou lève ValueError."""
    if name not in ROLES:
        choices = ", ".join(sorted(ROLES))
        raise ValueError(f"Rôle inconnu : '{name}'. Rôles disponibles : {choices}")
    return ROLES[name]


def load_project_context(root: Path) -> str:
    """Lit AGENTS.md à la racine du projet s'il existe."""
    agents_md = root.resolve() / "AGENTS.md"
    if not agents_md.is_file():
        return ""
    try:
        return agents_md.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def build_system_prompt(role: RoleSpec, root: Path) -> str:
    """Assemble le prompt système du rôle avec le contexte projet (AGENTS.md)."""
    context = load_project_context(root)
    if not context:
        return role.system_prompt
    return f"{role.system_prompt}\n\n--- CONTEXTE DU PROJET (AGENTS.md) ---\n{context}"
