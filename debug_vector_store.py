from qdrant_client import QdrantClient, models
from dotenv import load_dotenv; load_dotenv()
import os, textwrap

cli  = QdrantClient(url=os.getenv("QDRANT_URL"),
                    api_key=os.getenv("QDRANT_API_KEY"))
coll = os.getenv("QDRANT_COLLECTION")

def grep(word, limit=5):
    f = models.Filter(must=[
        models.FieldCondition(
            key="text",
            match=models.MatchText(text=word)
        )
    ])
    hits, _ = cli.scroll(coll, scroll_filter=f,
                         with_payload=True, limit=limit)
    print(f"\n🔍 '{word}' → {len(hits)} hit(s)")
    for h in hits:
        snippet = textwrap.shorten(
            h.payload["text"], 120, placeholder=" …")
        print("—", snippet)

grep("মামা")         # ভাগ্য দেবতা
grep("শুম্ভুনাথ")    # সুপুরুষ
grep("১৫")           # কল্যাণীর বয়স
