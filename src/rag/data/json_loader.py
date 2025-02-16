import os
import re
import json


class JsonLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = self.load_data()

    def load_data(self):
        docs = []
        for i, doc in enumerate(os.listdir(self.data_path)):
            if doc.endswith(".json"):
                file_path = os.path.join(self.data_path, doc)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = json.load(f)

                docs.append({
                    "id": str(i),
                    "filename": doc,
                    "text": self.clean_text(str(text))
                })
        return docs
    
    def clean_text(self, text):
        text = text.lower()

        # remove punctuation
        text = re.sub(r"[^\w\s]", "", text)

        # text = text.split(" ")
        return text
    