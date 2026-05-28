from pathlib import Path
from email_rag.config import Config, load_config, write_starter_config


def test_defaults():
    c = Config()
    assert c.imap_port == 993
    assert c.embed_model == "nomic-embed-text"
    assert c.answer_model == "claude-sonnet-4-6"
    assert c.top_k == 20
    assert c.folders == ["INBOX", "Sent"]


def test_write_and_load_roundtrip(tmp_path):
    path = tmp_path / "config.toml"
    write_starter_config(path, imap_host="imap.ebi.ac.uk", imap_user="ochoa@ebi.ac.uk")
    assert path.exists()
    c = load_config(path)
    assert c.imap_host == "imap.ebi.ac.uk"
    assert c.imap_user == "ochoa@ebi.ac.uk"
    assert c.folders == ["INBOX", "Sent"]


def test_load_missing_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml")
