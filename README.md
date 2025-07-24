# 📚 Bangla–English RAG System (HSC26 1st Paper)

A simple but powerful Retrieval-Augmented Generation (RAG) system that answers Bengali and English queries grounded in the *HSC26 Bangla 1st Paper* textbook using OCR, Gemini embeddings, Qdrant vector store, and a hybrid retrieval strategy (vector + BM25). 

> ⚡ No GPU required – everything runs on CPU and APIs (Gemini + Qdrant Cloud)

---

## ✅ Features

- 🔍 Accepts **Bangla** and **English** natural language questions
- 🧠 Combines **vector similarity** (semantic) and **BM25** (keyword) retrieval
- 🧾 Powered by **Google Gemini Pro** via LangChain
- 🔡 Fully functional **OCR-based document ingestion** and chunking
- 🗃️ Vector storage via **Qdrant Cloud**
- 🧠 **Short-term memory** for ongoing chat history
- 🔗 Lightweight REST API using FastAPI

---

## 🚀 Quick Start

```bash

# 1️⃣ Set up virtual env
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Configure secrets
cp .env.example .env  # then fill in your API keys and file paths

# 4️⃣ Create the vector index (run once)
python index.py

# 5️⃣ Start the API server
uvicorn main:app --reload
```

Now go to: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for Swagger UI.

---

## 🧠 Sample Queries

```text
Q: অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
 A: উত্তর পাইনি।  উদ্দীপকের কোথাও অনুপম কাকে সুপুরুষ বলে উল্লেখ ক      করেছেন সে বিষয়ে কোন তথ্য নেই।

Q: কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
✅ A: মামাকে

Q: বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
✅ A: ১৫ বছর
```

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `index.py` | Extracts text from PDF via OCR (pytesseract), chunks, embeds, and uploads to Qdrant |
| `main.py`  | FastAPI backend with LangChain Conversational RAG chain |
| `.env.example` | Template with environment variables for API keys, file paths, collection name |
| `requirements.txt` | Python dependency list |
| `test_query.py` | Simple script to test the running API endpoint with a Bangla query |

---

## 📊 Technical Choices & Justification

### 1. **Text Extraction**
- **Tool:** `pdf2image` + `pytesseract` (with OpenCV pre-thresholding)
- **Why:** The original PDF was image-based (scanned textbook), so OCR was required. Bengali script support with `lang="ben+eng"` improves fidelity.
- **Challenges:** Minor OCR noise and spacing issues, fixed using regex-based cleanup.

### 2. **Chunking Strategy**
- **Method:** RecursiveCharacterTextSplitter  
- **Settings:** `chunk_size=500`, `chunk_overlap=50`, `separators=["\n\n", "\n", "।", ".", " "]`
- **Why:** Bengali sentences are often long; this strategy ensures semantically intact and overlapping chunks for context continuity.

### 3. **Embeddings**
- **Model:** `GoogleGenerativeAIEmbeddings` (`embedding-001`)
- **Why:** Gemini's multilingual support and better Bangla text representation outperforms many alternatives.

### 4. **Vector Store & Similarity**
- **Backend:** `Qdrant Cloud` (Cosine distance, 768-d vectors)
- **Why:** Free, fast, hosted, and compatible with LangChain.  
- **Hybrid Retrieval:** Combines vector similarity with `BM25Retriever` (token-based search) using LangChain’s `EnsembleRetriever`.

### 5. **LLM Reasoning**
- **Model:** `gemini-1.5-flash`
- **Answer Prompt:** Grounded, fallback-aware prompt
- **Follow-up Prompt:** Self-contained rewriter for ambiguous chat input
- **Why:** Fast, cost-effective, and Bangla-aware LLM

### 6. **Evaluation**
-  **Groundedness:** All answers are generated based on retrieved chunks. Prompt design ensures this.
-  **Relevance:** BM25 helps recover OCR-token-aligned facts, and vector search adds semantic depth.


---

## 📌 Environment Variables (`.env`)

```env
GOOGLE_API_KEY=Your Gemini api key
QDRANT_URL=https://your-qdrant-url
QDRANT_API_KEY=your-qdrant-key
QDRANT_COLLECTION=hsc26_bangla
PDF_FILE=HSC26-Bangla1st-Paper.pdf

```

---

## 🧪 Sample Evaluation Strategy

| Metric         | Method                                    | Notes |
|----------------|-------------------------------------------|-------|
| Groundedness   | Check if answer spans exist in sources    | ✅ Done |
| Relevance      | Visual inspection + chunk preview logging | ✅ Good retrieval balance |
| Chunk Accuracy | Based on logical sentence breaks          | ✅ Recursive splitter worked well |

---

## 📮 API Documentation

```
POST /ask
{
  "question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
}
```

**Response:**
```json
{
  "answer": "শুম্ভুনাথ",
  "sources": [
    {
      "id": "some-uuid",
      "preview": "… শুম্ভুনাথ ছিল অনুপমের মতে সুপুরুষ …"
    }
  ]
}
```

---

## 🔧 Tools Used

- `LangChain 0.2.2`
- `FastAPI + Uvicorn`
- `Google Gemini Pro + Embeddings`
- `Qdrant (Cloud)`
- `Tesseract OCR + OpenCV`
- `pdf2image`, `pytesseract`
- `dotenv`, `requests`

---

## 📎 Future Improvements

- ✅ Use RAGAS or human labels for systematic evaluation
- ⏳ Fine-tune chunk sizes dynamically based on actual sentence length
- 🧩 Integrate with Langflow UI or Dify for GUI experience
- 🌐 Add English ↔ Bangla translation for even more flexibility

---

## 👨‍💻 Author

Md Mahabube Alahi Atik  