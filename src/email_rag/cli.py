import json as jsonlib
from datetime import datetime
from getpass import getpass
from pathlib import Path

import click

from email_rag.answerer import Answerer
from email_rag.config import DEFAULT_CONFIG_PATH, load_config, write_starter_config
from email_rag.db import Database
from email_rag.embedder import Embedder
from email_rag.fetcher import Fetcher
from email_rag.indexer import Indexer
from email_rag.parsing import parse_message
from email_rag.secrets import get_imap_password, set_imap_password
from email_rag.store import Filters, VectorStore


@click.group()
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG_PATH),
              help="Path to config.toml")
@click.pass_context
def main(ctx, config_path):
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path


def _load(ctx):
    return load_config(ctx.obj["config_path"])


@main.command()
@click.option("--imap-host", prompt=True)
@click.option("--imap-user", prompt=True)
@click.option("--imap-port", default=993, show_default=True, type=int)
@click.pass_context
def init(ctx, imap_host, imap_user, imap_port):
    """Write config, store IMAP password, verify IMAP login + Ollama."""
    path = ctx.obj["config_path"]
    cfg = write_starter_config(
        path, imap_host=imap_host, imap_user=imap_user, imap_port=imap_port
    )
    Database(cfg.db_path).init_schema(dim=768)
    pw = getpass("IMAP password: ")
    set_imap_password(imap_user, pw)
    try:
        Fetcher(cfg.imap_host, cfg.imap_port, cfg.imap_user, pw).check_login()
        click.echo("IMAP login OK.")
    except Exception as e:  # noqa: BLE001
        click.echo(f"WARNING: IMAP login failed: {e}", err=True)
    try:
        Embedder(cfg.ollama_url, cfg.embed_model).health_check()
        click.echo("Ollama OK.")
    except Exception as e:  # noqa: BLE001
        click.echo(f"WARNING: Ollama check failed: {e}", err=True)
    click.echo(f"Config written to {path}. DB at {cfg.db_path}.")


@main.command()
@click.option("--since", default=None, help="ISO date override for first run")
@click.option("--folders", default=None, help="Comma-separated folder override")
@click.option("--full", is_flag=True, help="Ignore stored UID state; rescan")
@click.pass_context
def sync(ctx, since, folders, full):
    """Fetch new mail over IMAP, parse, chunk, and embed."""
    cfg = _load(ctx)
    db = Database(cfg.db_path)
    db.init_schema(dim=768)
    _check_embed_model(db, cfg)
    embedder = Embedder(cfg.ollama_url, cfg.embed_model)
    embedder.health_check()
    store = VectorStore(db)
    indexer = Indexer(db, embedder, store)
    password = get_imap_password(cfg.imap_user)
    if not password:
        raise click.ClickException("No IMAP password stored. Run `email-rag init`.")
    fetcher = Fetcher(cfg.imap_host, cfg.imap_port, cfg.imap_user, password)

    folder_list = folders.split(",") if folders else cfg.folders
    since_date = _parse_since(since or cfg.since)

    for folder in folder_list:
        row = db.conn.execute(
            "SELECT uidvalidity, last_seen_uid FROM sync_state WHERE folder=?",
            (folder,),
        ).fetchone()
        last_uid = 0 if full else (row["last_seen_uid"] if row else 0)
        validity = None if full else (row["uidvalidity"] if row else None)
        click.echo(f"Syncing {folder} (from UID {last_uid + 1})...")
        batch: list = []
        skipped = 0
        for uid, raw in fetcher.sync_folder(folder, last_uid, validity, since_date):
            try:
                batch.append(parse_message(raw, folder=folder, uid=uid))
            except Exception as e:  # noqa: BLE001 — one bad message must not abort the run
                skipped += 1
                click.echo(f"  skipped uid {uid} in {folder}: {e}", err=True)
                continue
            if len(batch) >= 200:
                indexer.store_messages(batch)
                batch = []
        if batch:
            indexer.store_messages(batch)
        if skipped:
            click.echo(f"  ({skipped} unparseable message(s) skipped in {folder})")
        db.conn.execute(
            "INSERT OR REPLACE INTO sync_state(folder, uidvalidity, last_seen_uid)"
            " VALUES (?,?,?)",
            (folder, fetcher.last_uidvalidity, fetcher.last_uid),
        )
        db.conn.commit()

    click.echo("Embedding new chunks...")
    n = indexer.embed_pending()
    click.echo(f"Embedded {n} chunks.")


@main.command()
@click.argument("question")
@click.option("--from", "from_addr", default=None)
@click.option("--since", "since", default=None)
@click.option("--until", "until", default=None)
@click.option("-k", "top_k", type=int, default=None)
@click.option("--model", default=None)
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def ask(ctx, question, from_addr, since, until, top_k, model, as_json):
    """Answer a question over your email with citations."""
    import anthropic

    cfg = _load(ctx)
    if not Path(cfg.db_path).exists():
        raise click.ClickException("No index found. Run `email-rag sync` first.")
    db = Database(cfg.db_path)
    _check_embed_model(db, cfg)
    embedder = Embedder(cfg.ollama_url, cfg.embed_model)
    store = VectorStore(db)
    client = anthropic.Anthropic()
    answerer = Answerer(db, store, embedder, client, model or cfg.answer_model)
    filters = Filters(from_addr=from_addr, since=since, until=until)
    answer, sources = answerer.ask(question, k=top_k or cfg.top_k, filters=filters)

    if as_json:
        click.echo(jsonlib.dumps({"answer": answer, "sources": sources}, indent=2))
        return
    click.echo(answer)
    click.echo("\nSources:")
    for s in sources:
        click.echo(f"  [{s['n']}] {s['from']} · {s['date']} · {s['subject']} ({s['message_id']})")


@main.command()
@click.argument("query")
@click.option("--from", "from_addr", default=None)
@click.option("--since", "since", default=None)
@click.option("--until", "until", default=None)
@click.option("-k", "top_k", type=int, default=None)
@click.pass_context
def search(ctx, query, from_addr, since, until, top_k):
    """Retrieval only — ranked snippets, no LLM call."""
    cfg = _load(ctx)
    if not Path(cfg.db_path).exists():
        raise click.ClickException("No index found. Run `email-rag sync` first.")
    db = Database(cfg.db_path)
    _check_embed_model(db, cfg)
    embedder = Embedder(cfg.ollama_url, cfg.embed_model)
    store = VectorStore(db)
    filters = Filters(from_addr=from_addr, since=since, until=until)
    vec = embedder.embed_query(query)
    hits = store.hybrid_search(vec, query, k=top_k or cfg.top_k, filters=filters)
    for r in hits:
        click.echo(f"[{r.score:.4f}] {r.from_addr} · {r.date.date()} · {r.subject}")
        click.echo(f"    {r.text[:200].strip()}")


@main.command()
@click.pass_context
def status(ctx):
    """Show index stats."""
    cfg = _load(ctx)
    db = Database(cfg.db_path)
    db.init_schema(dim=768)
    n_msg = db.conn.execute("SELECT count(*) c FROM messages").fetchone()["c"]
    n_chunk = db.conn.execute("SELECT count(*) c FROM chunks").fetchone()["c"]
    n_pending = db.conn.execute(
        "SELECT count(*) c FROM chunks WHERE embedded=0"
    ).fetchone()["c"]
    click.echo(f"Messages: {n_msg}")
    click.echo(f"Chunks:   {n_chunk} ({n_pending} pending embedding)")
    click.echo(f"Embed model: {db.get_meta('embed_model') or cfg.embed_model}")
    for row in db.conn.execute("SELECT folder, last_seen_uid FROM sync_state"):
        click.echo(f"  {row['folder']}: last UID {row['last_seen_uid']}")


def _check_embed_model(db: Database, cfg) -> None:
    recorded = db.get_meta("embed_model")
    if recorded is None:
        db.set_meta("embed_model", cfg.embed_model)
    elif recorded != cfg.embed_model:
        raise click.ClickException(
            f"Index was built with embed model '{recorded}' but config says "
            f"'{cfg.embed_model}'. Queries must use the same model. "
            f"Re-index or restore the config model."
        )


def _parse_since(value):
    if not value:
        return None
    return datetime.fromisoformat(value).date()


if __name__ == "__main__":
    main()
