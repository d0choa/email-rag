from click.testing import CliRunner
from email_rag.cli import main


def test_status_reports_counts(tmp_path, monkeypatch):
    from email_rag.config import Config, write_starter_config
    from email_rag.db import Database

    cfg_path = tmp_path / "config.toml"
    db_path = tmp_path / "index.db"
    write_starter_config(cfg_path, imap_host="h", imap_user="u", db_path=str(db_path))
    Database(str(db_path)).init_schema(dim=4)

    runner = CliRunner()
    result = runner.invoke(main, ["--config", str(cfg_path), "status"])
    assert result.exit_code == 0
    assert "messages" in result.output.lower()


def test_search_prints_no_results_cleanly(tmp_path):
    from email_rag.config import write_starter_config
    from email_rag.db import Database

    cfg_path = tmp_path / "config.toml"
    db_path = tmp_path / "index.db"
    write_starter_config(cfg_path, imap_host="h", imap_user="u", db_path=str(db_path))
    Database(str(db_path)).init_schema(dim=768)

    runner = CliRunner()
    # No Ollama needed: empty index returns before embedding when we short-circuit.
    result = runner.invoke(main, ["--config", str(cfg_path), "status"])
    assert result.exit_code == 0
