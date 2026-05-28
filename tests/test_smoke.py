import os
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("EMAIL_RAG_LIVE") != "1",
    reason="set EMAIL_RAG_LIVE=1 to run live Ollama smoke test",
)


def test_ollama_embeds_768_dims():
    from email_rag.config import Config
    from email_rag.embedder import Embedder

    cfg = Config()
    emb = Embedder(cfg.ollama_url, cfg.embed_model)
    emb.health_check()
    vec = emb.embed_query("hello world")
    assert len(vec) == 768
