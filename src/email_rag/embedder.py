import httpx

# Safety net: never send an input longer than this to the model, regardless of
# chunking. ~2k tokens at 4 chars/token keeps us well within nomic-embed-text's
# context so a pathological chunk can't trigger a 400.
MAX_INPUT_CHARS = 8000


class Embedder:
    def __init__(self, url: str, model: str, client=None):
        self.url = url.rstrip("/")
        self.model = model
        self._client = client or httpx.Client(timeout=120.0)

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        resp = self._client.post(
            f"{self.url}/api/embed",
            json={"model": self.model, "input": inputs},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([f"search_document: {t[:MAX_INPUT_CHARS]}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([f"search_query: {text[:MAX_INPUT_CHARS]}"])[0]

    def health_check(self) -> None:
        """Raise if Ollama is unreachable or the model is missing."""
        resp = self._client.get(f"{self.url}/api/tags")
        resp.raise_for_status()
        models = {m["name"].split(":")[0] for m in resp.json().get("models", [])}
        if self.model.split(":")[0] not in models:
            raise RuntimeError(
                f"Ollama model '{self.model}' not found. Run: ollama pull {self.model}"
            )
