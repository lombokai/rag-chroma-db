import re

import numpy as np
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
        self.corpus = [text["text"] for text in data]

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
    
    def _ranking(self, ids: List[str], texts: List[str], scores: List[float]) -> List[dict]:
        ranked_result = []
        for id, text, score in zip(ids, texts, scores):
            result = {
                "ids": id,
                "texts": text,
                "scores": score
            }
            ranked_result.append(result)
        return ranked_result

    def bm25_search(self, query: str, top_n: int=10):
        tokenized_query = self._tokenize(query)
        scores = self.sparse.get_scores(tokenized_query)

        top_n_idx = scores.argsort()[-top_n:][::-1]

        data = [self.data[i] for i in top_n_idx]

        ids = [d["id"] for d in data]
        texts = [d["text"] for d in data]
        scores = [scores[i] for i in top_n_idx]

        ranked_result = self._ranking(ids, texts, scores)

        return ranked_result

    def transformer_search(self, query: str, top_n: int=10):
        query_embedding = self.embedder.embed(query)

        search_result = self.db.search_vectors(
            query = query_embedding,
            top_k = top_n
        )

        ids = [id for id in search_result["ids"][0]]
        texts = [text for text in search_result["documents"][0]]
        scores = [score for score in search_result["distances"][0]]

        ranked_result = self._ranking(ids, texts, scores)

        return ranked_result

    def rrf(self, rankings: List[List[int]], weights: List[float], k: int=60):
        rrf_scores = {}
        for weight, rank_list in zip(weights, rankings):
            for rank, doc_id in enumerate(rank_list):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + weight / (k + rank + 1)
        return rrf_scores

    def hybrid_search(
        self, query: str, top_n: int, sparse_weight: float, dense_weight:float
    ):
        query_embedding = self.embedder.embed(query)
        dense_result = self.db.search_vectors(
            query_embedding,
            top_k = top_n
        )
        dense_ids = [int(id) for id in dense_result["ids"][0]]

        tokenized_query = self._tokenize(query)
        sparse_scores = self.sparse.get_scores(tokenized_query)
        sparse_ids = np.argsort(sparse_scores)[::-1][:top_n]
        
        rankings = [dense_ids, sparse_ids.tolist()]
        weights = [dense_weight, sparse_weight]

        rrf_scores = self.rrf(rankings, weights)

        combined_ids = list(rrf_scores.keys())
        combined_scores = [rrf_scores[doc_id] for doc_id in combined_ids]
        combined_candidate = [self.data[id] for id in combined_ids]

        ranked_result = self._ranking(combined_ids, combined_candidate, combined_scores)
        
        text = []
        for res in ranked_result:
            text.append(res["text"]["text"])

        return text
