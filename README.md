# 🏥 Medical Literature Q&A System — RAG + PubMed

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://asifnawaz-medical-rag.streamlit.app)
[![PubMed](https://img.shields.io/badge/Data-PubMed-326599?style=for-the-badge)](https://pubmed.ncbi.nlm.nih.gov)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **RAG-powered Medical Q&A system** that retrieves relevant PubMed research papers and generates evidence-based clinical answers using Llama 3.3-70B — with automatic citations.

---

## 🎯 Problem Statement

Medical professionals and researchers spend hours searching through millions of papers to find evidence-based answers. This system automates that process — ask any clinical question and get an answer backed by real PubMed research papers with citations in seconds.

---

## 🚀 Live Demo

**[🔬 Try the App →](https://asifnawaz-medical-rag.streamlit.app)**

Example questions:
- *"What are the latest treatments for Type 2 diabetes?"*
- *"How effective is immunotherapy for cancer treatment?"*
- *"What are the long-term effects of COVID-19?"*
- *"What is the best treatment for hypertension?"*

---

## 🏗️ Architecture

```
User Question
      ↓
Sentence Transformer (all-MiniLM-L6-v2)
      ↓
FAISS Vector Search (385 PubMed papers)
      ↓
Top-K Relevant Papers Retrieved
      ↓
Llama 3.3-70B via Groq API
      ↓
Evidence-Based Answer + Citations
```

---

## 📊 System Specifications

| Component | Details |
|-----------|---------|
| Papers | 385 PubMed papers (2020–2024) |
| Topics | Diabetes, Cancer, COVID-19, Hypertension, CVD, Mental Health, Obesity, Antibiotics |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Dimensions | 384 |
| Vector Database | FAISS (IndexFlatIP — cosine similarity) |
| LLM | Llama 3.3-70B via Groq API |
| Frontend | Streamlit |

---

## 🗂️ Project Structure

```
medical-rag-pubmed/
│
├── data/
│   ├── raw/
│   │   └── pubmed_papers.csv          # 385 PubMed papers
│   └── processed/
│       ├── pubmed.index               # FAISS vector index
│       ├── metadata.pkl               # Paper metadata
│       └── config.pkl                 # System config
│
├── src/
│   ├── fetch_pubmed.py                # PubMed API data pipeline
│   ├── build_index.py                 # FAISS index builder
│   └── rag_pipeline.py                # RAG chain + retrieval
│
├── dashboard/
│   └── streamlit_app.py              # Live web application
│
├── notebooks/
│   └── 01_data_exploration.ipynb     # Data analysis
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Data Source | PubMed API (NCBI Entrez) |
| Embeddings | sentence-transformers |
| Vector DB | FAISS (Facebook AI) |
| LLM | Llama 3.3-70B via Groq |
| Framework | LangChain |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud |

---

## 🚀 Run Locally

```bash
# Clone
git clone https://github.com/Asif5588-M/medical-rag-pubmed.git
cd medical-rag-pubmed

# Environment
conda create -n rag-env python=3.11 -y
conda activate rag-env
pip install -r requirements.txt

# Add Groq API key
echo 'GROQ_API_KEY=your_key_here' > .env

# Fetch papers + build index
python src/fetch_pubmed.py
python src/build_index.py

# Run app
streamlit run dashboard/streamlit_app.py
```

---

## 💡 Use Cases

- Clinical decision support for doctors
- Medical literature review automation
- Healthcare research assistance
- Drug and treatment information retrieval
- Patient education content generation

---

## 👨‍💻 Author

**Asif Nawaz**
- 🏥 Medical Data Scientist | PMAS Arid Agriculture University
- 🎓 MPhil Economics (Health Economics)
- 📄 Published Researcher — HEC Y-Category Journal
- 🌐 [Pakistan CHE Dashboard](https://asifnawaz-pakistan-health.streamlit.app)
- 💊 [Drug Sentiment Analyzer](https://asifnawaz-drug-sentiment.streamlit.app)
- 🔬 [Medical RAG System](https://asifnawaz-medical-rag.streamlit.app)
- 🤗 [HuggingFace](https://huggingface.co/asif-nawaz-ml)
- 🔗 [GitHub](https://github.com/Asif5588-M)