from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ParsedMessage:
    message_id: str
    from_addr: str
    to_addrs: list[str]
    cc_addrs: list[str]
    subject: str
    date: datetime
    body: str
    in_reply_to: str | None = None
    references: list[str] = field(default_factory=list)
    attachment_names: list[str] = field(default_factory=list)
    folder: str = ""
    uid: int = 0
    thread_id: str | None = None


@dataclass
class Chunk:
    message_id: str
    ord: int
    text: str
    chunk_id: int | None = None


@dataclass
class Retrieved:
    chunk_id: int
    message_id: str
    text: str
    score: float
    from_addr: str
    subject: str
    date: datetime
    thread_id: str
