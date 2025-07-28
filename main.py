# main.py — hybrid RAG (Qdrant vectors + BM25 keyword)  •  LangChain 0.2.2
# -----------------------------------------------------------------------
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# ── 0. environment ───────────────────────────────────────────────────────────
load_dotenv()

# ── 1. embeddings & vector store ─────────────────────────────────────────────
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=os.getenv("QDRANT_COLLECTION"),
    embedding=embeddings,
    content_payload_key="text",
)

# ── 2. hybrid retriever (vectors + BM25) ─────────────────────────────────────
vect_ret = vector_store.as_retriever(search_kwargs={"k": 12})  # ↑k → better recall

docs, offset = [], None
while True:                                   # pull every doc once for BM25
    hits, offset = client.scroll(
        os.getenv("QDRANT_COLLECTION"),
        offset=offset,
        with_payload=True,
        limit=512,
    )
    docs += [
        Document(
            page_content=h.payload["text"],
            metadata={
                "id": str(h.id),
                "preview": h.payload["text"][:40]
            },                               # ← no more empty {}
        )
        for h in hits
        if "text" in h.payload
    ]
    if offset is None:
        break

bm25_ret = BM25Retriever.from_documents(docs)
bm25_ret.k = 8

retriever = EnsembleRetriever(
    retrievers=[vect_ret, bm25_ret],
    weights=[0.6, 0.4],          # ↑vector weight, ↓BM25 to damp OCR noise
)

# ── 3. LLMs & chains ─────────────────────────────────────────────────────────
llm_answer   = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4)
llm_condense = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# 3a. answering chain (grounded prompt)
answer_prompt = PromptTemplate.from_template(
    "প্রসঙ্গ (শুধু নিচের তথ্য ব্যবহার করুন):\n{context}\n\n"
    "প্রশ্ন: {question}\n\n"
    "উত্তর (উদ্ধৃতি সহ লিখুন, যদি সরাসরি তথ্য না থাকে, তবে অনুমানযোগ্য হলে উত্তর দিন, নইলে 'উত্তর পাইনি' বলুন।):"
)
answer_chain = StuffDocumentsChain(
    llm_chain=LLMChain(llm=llm_answer, prompt=answer_prompt),
    document_variable_name="context",
)

# 3b. follow-up question re-writer
condense_prompt = PromptTemplate.from_template(
    "পূর্ববর্তী আলাপ:\n{chat_history}\n\nবর্তমান প্রশ্ন: {question}\n"
    "শুধু তথ্যপূর্ণ, স্বয়ংসম্পূর্ণ প্রশ্ন তৈরি করুন (বাংলায়):"
)
question_generator = LLMChain(llm=llm_condense, prompt=condense_prompt)

# 3c. short-term memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    output_key="answer",
    return_messages=True,
)

# 3d. conversational RAG chain
rag_chain = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator,
    combine_docs_chain=answer_chain,
    memory=memory,
    output_key="answer",
    return_source_documents=True,
)

# ── 4. FastAPI wrapper ───────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str

app = FastAPI(title="Bangla–English RAG (Hybrid)")

@app.post("/ask")
async def ask(payload: AskRequest):
    q = payload.question.strip()
    if not q:
        raise HTTPException(400, "Question cannot be empty.")
    result = rag_chain.invoke({"question": q})
    return {
        "answer":  result["answer"],
        "sources": [d.metadata for d in result["source_documents"]],
    }

@app.get("/")
def root():
    return {"status": "ok", "message": "Bangla RAG hybrid ready "}
