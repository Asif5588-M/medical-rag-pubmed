import faiss
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

PROCESSED = Path("data/processed")

# ── Load Index + Metadata ──────────────────────────────────
def load_rag_system():
    print("Loading RAG system...")

    index = faiss.read_index(str(PROCESSED / "pubmed.index"))

    with open(PROCESSED / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=1024,
    )

    print("RAG system ready ✓")
    return index, metadata, embedder, llm


# ── Retrieve Relevant Papers ───────────────────────────────
def retrieve(query: str, index, metadata, embedder, top_k: int = 5):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, ids = index.search(q_emb.astype(np.float32), k=top_k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < len(metadata):
            paper = metadata[idx].copy()
            paper["score"] = float(score)
            results.append(paper)
    return results


# ── Generate Answer ────────────────────────────────────────
PROMPT = ChatPromptTemplate.from_template("""
You are a medical expert AI assistant. Answer the clinical question based ONLY on the provided research papers.

RESEARCH PAPERS:
{context}

CLINICAL QUESTION: {question}

Instructions:
- Give a clear, evidence-based answer
- Cite papers using [1], [2] etc
- Mention limitations if any
- Keep answer under 300 words
- End with "Sources:" listing the papers used

ANSWER:
""")


def answer(query: str, index, metadata, embedder, llm, top_k: int = 5):
    papers = retrieve(query, index, metadata, embedder, top_k)

    context = ""
    for i, p in enumerate(papers, 1):
        context += f"""
[{i}] Title: {p['title']}
     Journal: {p['journal']} ({p['year']})
     Authors: {p['authors']}
     Abstract: {p['abstract'][:400]}...
     URL: {p['url']}
"""

    chain    = PROMPT | llm
    response = chain.invoke({
        "context" : context,
        "question": query,
    })

    return response.content, papers


# ── Test ───────────────────────────────────────────────────
if __name__ == "__main__":
    index, metadata, embedder, llm = load_rag_system()

    test_questions = [
        "What are the latest treatments for Type 2 diabetes?",
        "How effective is immunotherapy for cancer treatment?",
        "What are the long-term effects of COVID-19?",
    ]

    print("\n" + "="*60)
    print("RAG PIPELINE TEST")
    print("="*60)

    for q in test_questions:
        print(f"\nQ: {q}")
        print("-"*50)
        ans, papers = answer(q, index, metadata, embedder, llm)
        print(ans)
        print(f"\nRetrieved {len(papers)} papers")
        print("="*60)

    print("\nNext → dashboard/streamlit_app.py")