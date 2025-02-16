import os


class DataManager:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self):
        docs = []
        for doc in self.data_path:
            if doc.endswith(".txt"):
                file_path = os.path.join(self.data_path, doc)
                with open(file_path, "r", encoding="utf-8") as f:
                    docs.append({
                        "id": doc,
                        "text": f.read()
                    })
        return docs

    def split_text(self, text, chunk_size=1000, overlap=20):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def chunk_text(self):
        chunked_data = []
        for doc in self.load_data():
            chunks = self.split_text(doc["text"])
            for chunk in chunks:
                chunked_data.append({
                    "id": doc["id"],
                    "text": chunk
                })
        return chunked_data
