import tomllib
from dataclasses import dataclass, asdict, field
from pathlib import Path

import tomli_w

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "email-rag" / "config.toml"
DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "email-rag" / "index.db"


@dataclass
class Config:
    imap_host: str = ""
    imap_port: int = 993
    imap_user: str = ""
    folders: list[str] = field(default_factory=lambda: ["INBOX", "Sent"])
    since: str = ""  # ISO date (YYYY-MM-DD) for first-run window; "" = all
    embed_model: str = "nomic-embed-text"
    answer_model: str = "claude-sonnet-4-6"
    ollama_url: str = "http://localhost:11434"
    db_path: str = str(DEFAULT_DB_PATH)
    top_k: int = 20


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No config at {path}. Run `email-rag init`.")
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return Config(**{k: v for k, v in data.items() if k in Config.__dataclass_fields__})


def write_starter_config(path: Path, **overrides) -> Config:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = Config(**overrides)
    with open(path, "wb") as f:
        tomli_w.dump(asdict(cfg), f)
    return cfg
