"""Tests unitaires pour harness.tools (garde-fous, édition ciblée, permissions)."""

from pathlib import Path

from harness.tools import ToolBox


def test_list_and_read_file_with_line_range(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("line1\nline2\nline3\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "ignored.py").write_text("x", encoding="utf-8")
    (tmp_path / "demo.egg-info").mkdir()
    (tmp_path / "demo.egg-info" / "PKG-INFO").write_text("x", encoding="utf-8")

    box = ToolBox(tmp_path, {"list_files", "read_file"})

    listing = box.execute("list_files", {})
    assert listing == "src/app.py"

    content = box.execute("read_file", {"path": "src/app.py", "start_line": 2, "end_line": 3})
    assert content == "2: line2\n3: line3"


def test_edit_file_replaces_unique_target_and_rejects_ambiguous(tmp_path: Path) -> None:
    target_file = tmp_path / "sample.py"
    target_file.write_text("a = 1\nb = 2\na = 1\n", encoding="utf-8")
    box = ToolBox(tmp_path, {"edit_file"})

    ambiguous = box.execute("edit_file", {"path": "sample.py", "target": "a = 1", "replacement": "a = 9"})
    assert "ambigu" in ambiguous
    assert target_file.read_text(encoding="utf-8") == "a = 1\nb = 2\na = 1\n"

    missing = box.execute("edit_file", {"path": "sample.py", "target": "c = 3", "replacement": "c = 4"})
    assert "introuvable" in missing

    ok = box.execute("edit_file", {"path": "sample.py", "target": "b = 2", "replacement": "b = 42"})
    assert "succès" in ok
    assert target_file.read_text(encoding="utf-8") == "a = 1\nb = 42\na = 1\n"


def test_toolbox_enforces_allowed_tools_and_write_prefixes(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "code.py").write_text("x = 1\n", encoding="utf-8")

    # Rôle 'test' : écriture autorisée uniquement dans tests/.
    test_box = ToolBox(
        tmp_path,
        {"read_file", "write_file", "edit_file"},
        allowed_write_prefixes=("tests/",),
    )

    denied_tool = test_box.execute("run_checks", {})
    assert "non autorisé" in denied_tool

    denied_write = test_box.execute("write_file", {"path": "src/code.py", "content": "hack"})
    assert "Écriture refusée" in denied_write
    assert (tmp_path / "src" / "code.py").read_text(encoding="utf-8") == "x = 1\n"

    allowed_write = test_box.execute("write_file", {"path": "tests/test_code.py", "content": "assert True\n"})
    assert "succès" in allowed_write
    assert (tmp_path / "tests" / "test_code.py").is_file()


def test_path_traversal_is_blocked(tmp_path: Path) -> None:
    box = ToolBox(tmp_path, {"read_file", "write_file"})

    res = box.execute("read_file", {"path": "../secret.txt"})
    assert "Accès refusé" in res


def test_schemas_only_expose_allowed_tools(tmp_path: Path) -> None:
    box = ToolBox(tmp_path, {"list_files", "read_file", "git_diff"})
    names = {schema["function"]["name"] for schema in box.schemas()}
    assert names == {"list_files", "read_file", "git_diff"}


def test_sensitive_files_and_ignored_dirs_are_hidden_and_blocked(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / ".env").write_text("GEMINI_API_KEY=secret\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text("SECRET=1\n", encoding="utf-8")
    (tmp_path / "certs").mkdir()
    (tmp_path / "certs" / "server.key").write_text("PRIVATE KEY", encoding="utf-8")
    (tmp_path / "certs" / "cert.pem").write_text("PEM DATA", encoding="utf-8")
    (tmp_path / "id_rsa").write_text("SSH KEY", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")

    box = ToolBox(tmp_path, {"list_files", "read_file", "write_file"})

    assert box.execute("list_files", {}) == "app.py"

    for blocked_path in (
        ".env",
        ".env.local",
        "certs/server.key",
        "certs/cert.pem",
        "id_rsa",
        ".git/config",
        ".venv/pyvenv.cfg",
    ):
        assert "Accès refusé" in box.execute("read_file", {"path": blocked_path})

    assert "Accès refusé" in box.execute("list_files", {"directory": ".git"})
    assert "Accès refusé" in box.execute("list_files", {"directory": ".venv"})
    assert "Accès refusé" in box.execute("write_file", {"path": ".env", "content": "X=1"})


def test_list_files_handles_none_and_empty_directory(tmp_path: Path) -> None:
    (tmp_path / "readme.md").write_text("# Demo\n", encoding="utf-8")
    box = ToolBox(tmp_path, {"list_files"})

    assert box.execute("list_files", {"directory": None}) == "readme.md"
    assert box.execute("list_files", {"directory": ""}) == "readme.md"


def test_non_string_tool_arguments_are_rejected(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("x = 1\n", encoding="utf-8")
    box = ToolBox(tmp_path, {"list_files", "read_file", "write_file", "edit_file", "git_diff"})

    assert "invalide" in box.execute("write_file", {"path": "app.py", "content": None})
    assert target.read_text(encoding="utf-8") == "x = 1\n"

    assert "invalide" in box.execute("write_file", {"path": None, "content": "x = 2\n"})
    assert not (tmp_path / "None").exists()

    assert "invalide" in box.execute(
        "edit_file", {"path": "app.py", "target": "x = 1", "replacement": None}
    )
    assert target.read_text(encoding="utf-8") == "x = 1\n"

    assert "invalide" in box.execute("read_file", {"path": 123})
    assert "invalide" in box.execute("list_files", {"directory": ["src"]})
    assert "invalide" in box.execute("git_diff", {"path": 42})


def test_execute_with_status_tracks_mutations_without_string_heuristics(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    box = ToolBox(tmp_path, {"edit_file", "write_file", "read_file"})

    # Même si le nom du fichier contient "succès", un échec d'édition ne doit pas marquer mutated=True.
    missing = box.execute_with_status(
        "edit_file", {"path": "src/succès.py", "target": "x", "replacement": "y"}
    )
    assert not missing.mutated

    written = box.execute_with_status("write_file", {"path": "src/succès.py", "content": "a = 1\n"})
    assert written.mutated

    read_res = box.execute_with_status("read_file", {"path": "src/succès.py"})
    assert not read_res.mutated

    failed_edit = box.execute_with_status(
        "edit_file", {"path": "src/succès.py", "target": "absent", "replacement": "b = 2"}
    )
    assert not failed_edit.mutated

    ok_edit = box.execute_with_status(
        "edit_file", {"path": "src/succès.py", "target": "a = 1", "replacement": "a = 2"}
    )
    assert ok_edit.mutated
