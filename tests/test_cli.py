"""Tests for EngramVault CLI commands."""

from pathlib import Path
from unittest.mock import patch
import pytest

from engramvault.cli.main import main
from engramvault.db.engine import Database


def test_cli_init_and_stats(tmp_path: Path, capsys):
    db_file = tmp_path / "cli_vault.db"
    
    # Test init
    with patch("sys.argv", ["memoria", "--db", str(db_file), "init"]):
        main()
    out, _ = capsys.readouterr()
    assert "Database inizializzato correttamente" in out
    assert db_file.exists()

    # Test stats
    with patch("sys.argv", ["memoria", "--db", str(db_file), "stats"]):
        main()
    out, _ = capsys.readouterr()
    assert "Statistiche Vault" in out
    assert "conversations" in out


def test_cli_seed_and_inspect(tmp_path: Path, capsys):
    db_file = tmp_path / "cli_seed.db"
    
    # Test seed
    with patch("sys.argv", ["memoria", "--db", str(db_file), "seed"]):
        main()
    out, _ = capsys.readouterr()
    assert "Inserite" in out
    assert "conversazioni di esempio" in out

    # Test entities
    with patch("sys.argv", ["memoria", "--db", str(db_file), "entities"]):
        main()
    out, _ = capsys.readouterr()
    assert "Catalogo Entità" in out

    # Test decisions
    with patch("sys.argv", ["memoria", "--db", str(db_file), "decisions"]):
        main()
    out, _ = capsys.readouterr()
    assert "Registro delle Decisioni" in out

    # Test timeline
    with patch("sys.argv", ["memoria", "--db", str(db_file), "timeline"]):
        main()
    out, _ = capsys.readouterr()
    assert "Timeline Eventi & Setup" in out

    # Test search
    with patch("sys.argv", ["memoria", "--db", str(db_file), "search", "cassette"]):
        main()
    out, _ = capsys.readouterr()
    assert "Risultati della Ricerca Ibrida per:" in out

    # Test ask
    with patch("sys.argv", ["memoria", "--db", str(db_file), "ask", "Cosa possiedo?", "--mode", "structured"]):
        main()
    out, _ = capsys.readouterr()
    assert "DOMANDA: Cosa possiedo?" in out
    assert "MODALITÀ ESEGUITA: [STRUCTURED]" in out


def test_cli_sync_exclude_flags(tmp_path: Path, capsys):
    ignore_file = tmp_path / "sync_ignore.txt"
    ignore_file.write_text("*.secret\n/private/path/*\n")

    # Test sync claude-code with --dry-run, --exclude, --ignore-file
    with patch("engramvault.sync.claude_code.ClaudeCodeSyncer.sync", return_value={"status": "dry_run", "changed_count": 0}) as mock_sync:
        with patch("sys.argv", ["memoria", "sync", "claude-code", "--dry-run", "--exclude", "*bank*,*salary*", "--ignore-file", str(ignore_file)]):
            main()
        assert mock_sync.called
    out, _ = capsys.readouterr()
    assert "[DRY-RUN]" in out

    # Test sync antigravity with --dry-run, --exclude, --ignore-file
    with patch("engramvault.sync.antigravity.AntigravitySyncer.sync", return_value={"status": "dry_run", "changed_count": 0}) as mock_sync_anti:
        with patch("sys.argv", ["memoria", "sync", "antigravity", "--dry-run", "--exclude", "*classified*", "--ignore-file", str(ignore_file)]):
            main()
        assert mock_sync_anti.called
    out, _ = capsys.readouterr()
    assert "[DRY-RUN]" in out

