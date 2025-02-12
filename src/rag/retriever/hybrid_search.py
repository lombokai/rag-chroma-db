import re
from typing import List
from rank_bm25 import BM25Okapi

from rag.vector_db import ChromaDB
from rag.vectorizer import Embedder


class HybridSearch:
    def __init__(
        self, 
        data: dict, 
        embedder_path: str,
        collection_name: str,
        db: ChromaDB,
        embedder: Embedder
    ):

        self.data = data
        self.corpus = [text for text in data["text"]]

        self.tokenized = list(map(self._tokenize, self.corpus))

        self.sparse = BM25Okapi(self.tokenized)

        self.embedder = embedder(embedder_path)
        self.db = db(collection_name)

    def _tokenize(self, text: str) -> list:
        text = text.lower()

        # remove punctuation
        text = re.sub(r"[^\w\s]", "", text)

        text = text.split(" ")
        return text
    
    def _ranking(self, candidates: List[str], scores: List[float]) -> List[dict]:
        ranked_result = []
        for candidate, score in zip(candidates, scores):
            result = {
                "candidates": candidate,
                "scores": score
            }
            ranked_result.append(result)
        return ranked_result

    def bm25_search(self, query: str, top_n: int=10):
        tokenized_query = self._tokenize(query)
        scores = self.sparse.get_scores(tokenized_query)

        top_n_idx = scores.argsort()[-top_n:][::-1]

        texts = [self.corpus[i] for i in top_n_idx]
        scores = [scores[i] for i in top_n_idx]

        ranked_result = self._ranking(texts, scores)

        return ranked_result

    def transformer_search(self, query: str, top_n: int=10):
        query_embedding = self.embedder.embed(query)

        search_result = self.db.search_vectors(
            query = query_embedding,
            top_k = top_n
        )

        candidates = [c for c in search_result["ids"][0]]
        scores = [s for s in search_result["distances"][0]]

        ranked_result = self._ranking(candidates, scores)

        return ranked_result

    def rrf(self):
        pass

    def hybrid_search(self):
        pass