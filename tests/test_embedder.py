import json
from email_rag.embedder import Embedder


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self):
        self.calls = []

    def post(self, url, json):
        self.calls.append((url, json))
        n = len(json["input"])
        return FakeResponse({"embeddings": [[0.1, 0.2, 0.3, 0.4] for _ in range(n)]})


def test_embed_documents_adds_prefix():
    client = FakeClient()
    emb = Embedder(url="http://x", model="nomic-embed-text", client=client)
    vecs = emb.embed_documents(["hello", "world"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 4
    sent_inputs = client.calls[0][1]["input"]
    assert sent_inputs == ["search_document: hello", "search_document: world"]


def test_embed_query_adds_prefix():
    client = FakeClient()
    emb = Embedder(url="http://x", model="nomic-embed-text", client=client)
    vec = emb.embed_query("what is up")
    assert len(vec) == 4
    assert client.calls[0][1]["input"] == ["search_query: what is up"]
