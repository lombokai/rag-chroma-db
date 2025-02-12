import chromadb
import logging

from rag.config import settings


logging.basicConfig(level=logging.INFO)


class ChromaDB:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.collection = None

        self._connect()
        self.create_collection()

    def _connect(self):
        try:
            self.client = chromadb.HttpClient(
                host=settings.CHROMA_HOST, 
                port=settings.CHROMA_PORT
            )
        except Exception as e:
            logging.info(f"Error connecting to ChromaDB: {e}")

    def create_collection(self):
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "ip",
                }
            )
        except Exception as e:
            logging.info(f"Error creating collection: {e}")
            raise e

    def insert_vectors(
        self, 
        ids: list, 
        docs: list,
        embeddings: list
    ):
        if not self.collection:
            raise ValueError("Collection not initialized")
        
        self.collection.add(
            ids = ids,
            documents = docs,
            embeddings = embeddings
        )

    def search_vectors(
        self,
        query: str,
        top_k: int
    ):
        if not self.collection:
            raise ValueError("Collection not initialized")
        
        results = self.collection.query(
            query_embeddings = [query],
            n_results = top_k
        )
        return results
    
    def delete_collection(self, vector_id: str):
        if not self.collection:
            raise ValueError("Collection not initialized")
        
        self.collection.delete(ids=[vector_id])