
# index.py  – OCR-only, preview, upload
# -------------------------------------
import os, re, uuid, textwrap, cv2, numpy as np
from pathlib import Path
from dotenv import load_dotenv
from pdf2image import convert_from_path            # pip install pdf2image
import pytesseract                                 # pip install pytesseract
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

# 🔧 tell the wrapper exactly where tesseract.exe is
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ──────────────────────────────────────────────────────────────────────────────

# 0. env ----------------------------------------------------------------------
load_dotenv()
PDF        = Path(os.getenv("PDF_FILE", "HSC26-Bangla1st-Paper.pdf"))
COLLECTION = os.getenv("QDRANT_COLLECTION", "hsc26_bangla")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_KEY = os.getenv("QDRANT_API_KEY")

POPPLER_BIN = r"C:\poppler-24.08.0\Library\bin"    # your Poppler path



if not PDF.exists():
    raise SystemExit(f" PDF not found: {PDF}")

# ── 1. helper functions ───────────────────────────────────────────────────────
CLEAN = re.compile(r"(Page\s+\d+|•|\s{2,})")


def clean(text: str) -> str:
    return CLEAN.sub(" ", text).strip()

def ocr_pil(pil_img):
    g = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    g = cv2.threshold(g, 0, 255,
                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return pytesseract.image_to_string(
        g, lang="ben+eng", config="--oem 3 --psm 6"
    )

# ── 2. OCR every page ─────────────────────────────────────────────────────────
print("  OCR-ing pages @500 dpi …")
texts = []
for page_img in convert_from_path(
        str(PDF), dpi=500, poppler_path=POPPLER_BIN):
    texts.append(clean(ocr_pil(page_img)))

full_text = "\n".join(texts)
print(f" OCR complete – {len(full_text):,} characters extracted.")

# ── 3. chunk & preview ────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50,
    separators=["\n\n", "\n", "।", ".", " "],
)
docs = splitter.create_documents([full_text])

print("\n  Preview first 3 chunks:")
for d in docs[:3]:
    print("—", textwrap.shorten(d.page_content, 200, placeholder=" …"))

if input("\nLooks good?  type YES to upload 👉 ").lower() != "yes":
    raise SystemExit("  Aborted – OCR needs tweaking.")

# ── 4. embed & upload to Qdrant ───────────────────────────────────────────────
print("🔗 Embedding with Gemini …")
emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

print(" Connecting to Qdrant …")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)

if client.collection_exists(COLLECTION):
    print("  Deleting old collection …")
    client.delete_collection(COLLECTION)

client.create_collection(
    collection_name=COLLECTION,
    vectors_config={"size": 768, "distance": "Cosine"},
)

ids      = [str(uuid.uuid4()) for _ in docs]
vectors  = [emb.embed_query(d.page_content) for d in docs]
payloads = [{"text": d.page_content} for d in docs]

print(f"  Uploading {len(ids)} vectors …")
client.upload_collection(
    collection_name=COLLECTION,
    vectors=vectors,
    payload=payloads,
    ids=ids,
)
print(f" Uploaded {len(ids)} vectors to '{COLLECTION}'")
