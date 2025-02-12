import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    CHROMA_HOST: str = os.getenv("CHROMA_HOST")
    CHROMA_PORT: int = os.getenv("CHROMA_PORT")


settings = Settings()
