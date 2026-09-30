"""Tests for Memoria AI CLI commands."""

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
