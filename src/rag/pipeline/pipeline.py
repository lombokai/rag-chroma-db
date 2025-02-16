from rag.retriever import HybridSearch
from rag.generator import TextGenerator
from rag.vector_db import ChromaDB
from rag.vectorizer import Embedder
from rag.data import JsonLoader


class RAGPipeline:
    def __init__(
        self,
        data_path: str,
        embedder_path: str,
        collection_name: str,
        api_key: str

    ):
        self.data_path = data_path
        self.load_data()

        self.embedder_path = embedder_path
        self.collection_name = collection_name
        self.load_retriever()

        self.api_key = api_key
        self.load_generator()

    def load_data(self):
        loader = JsonLoader(self.data_path)
        self.data = loader.data

    def load_retriever(self):
        self.retriever = HybridSearch(
            data=self.data,
            embedder_path=self.embedder_path,
            collection_name=self.collection_name,
            db=ChromaDB,
            embedder=Embedder
        )

    def load_generator(self):
        self.generator = TextGenerator(self.api_key)

    def run(self):
        while True:
            query = input()
            relevant_chunks = self.retriever.hybrid_search(
                query=query,
                top_n=5,
                sparse_weight=0.5,
                dense_weight=0.5
            )
            answer = self.generator.generate_response(query, relevant_chunks)
            
            print(answer)
            return answer
