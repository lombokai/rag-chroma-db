from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, embedder_path: str, dimensions: int=512):
        self.embedder_path = embedder_path
        self.dimensions = dimensions

        self.embedder = self._load_embedder()

    def _load_embedder(self):
        model = SentenceTransformer(
            self.embedder_path,
            truncate_dim=self.dimensions,
            device="cpu",
        )
        return model

    def embed(self, text):
        return self.embedder.encode(text)
