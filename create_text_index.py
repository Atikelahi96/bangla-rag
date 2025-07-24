# create_text_index.py
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import os

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

COLL = os.getenv("QDRANT_COLLECTION")

client.create_payload_index(
    collection_name=COLL,
    field_name="text",
    field_schema=models.TextIndexParams(
        type="text",                           # ← required
        tokenizer=models.TokenizerType.WORD,
        min_token_len=2,
        max_token_len=20,
    ),
)

print("✅ Text index for 'text' payload created.")
