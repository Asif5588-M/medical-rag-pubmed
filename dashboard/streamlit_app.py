import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag_pipeline import load_rag_system, answer

st.set_page_config(
    page_title="Medical RAG — PubMed Q&A",
    page_icon="🏥",
    layout="wide"
)

# ── Load System ────────────────────────────────────────────
@st.cache_resource
def get_rag():
    return load_rag_system()

# ── Header ─────────────────────────────────────────────────
st.title("🏥 Medical Literature Q&A System")
st.caption("RAG-powered · 385 PubMed papers · Llama 3.3-70B · By Asif Nawaz, PMAS Arid Agriculture University")
st.divider()

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Papers to retrieve", 3, 10, 5)
    st.divider()

    st.header("📊 System Info")
    st.metric("PubMed Papers", "385")
    st.metric("Topics Covered", "8")
    st.metric("LLM Model", "Llama 3.3-70B")
    st.metric("Embedding", "MiniLM-L6-v2")
    st.divider()

    st.header("💡 Example Questions")
    examples = [
        "What are the latest treatments for Type 2 diabetes?",
        "How effective is immunotherapy for cancer?",
        "What are long-term effects of COVID-19?",
        "What is the best treatment for hypertension?",
        "How does obesity affect cardiovascular disease?",
        "What are new approaches for depression treatment?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state["question"] = ex

# ── Main ───────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_area(
        "Ask a clinical question:",
        value=st.session_state.get("question", ""),
        height=120,
        placeholder="e.g. What are the latest treatments for Type 2 diabetes?"
    )

    ask = st.button("🔍 Search PubMed & Answer",
                    type="primary", use_container_width=True)

with col2:
    st.info("""
    **How it works:**
    1. Your question → embeddings
    2. FAISS searches 385 papers
    3. Top papers → Llama 3.3
    4. Evidence-based answer + citations
    """)

# ── Answer ─────────────────────────────────────────────────
if ask and question.strip():
    with st.spinner("Searching PubMed papers and generating answer..."):
        try:
            index, metadata, embedder, llm = get_rag()
            ans, papers = answer(
                question, index, metadata,
                embedder, llm, top_k=top_k
            )

            st.divider()

            # Answer box
            st.subheader("📋 Evidence-Based Answer")
            st.markdown(f"""
            <div style='background:var(--background-color);
                        border-left:4px solid #185FA5;
                        padding:16px 20px;
                        border-radius:0 8px 8px 0;
                        line-height:1.8;'>
            {ans.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

            st.divider()

            # Source papers
            st.subheader(f"📚 Retrieved Papers ({len(papers)})")
            for i, p in enumerate(papers, 1):
                with st.expander(
                    f"[{i}] {p['title'][:80]}... "
                    f"| {p['journal'][:30]} ({p['year']}) "
                    f"| Score: {p['score']:.3f}"
                ):
                    st.write(f"**Authors:** {p['authors']}")
                    st.write(f"**Journal:** {p['journal']}")
                    st.write(f"**Year:** {p['year']}")
                    st.write(f"**Abstract:** {p['abstract'][:500]}...")
                    st.markdown(f"**[Read Full Paper →]({p['url']})**")

        except Exception as e:
            st.error(f"Error: {e}")

elif ask and not question.strip():
    st.warning("Please enter a question first!")

st.divider()
st.caption(
    "Data: PubMed (2020-2024) · "
    "Embeddings: sentence-transformers · "
    "Vector DB: FAISS · "
    "LLM: Llama 3.3-70B via Groq · "
    "Asif Nawaz | PMAS Arid Agriculture University"
)