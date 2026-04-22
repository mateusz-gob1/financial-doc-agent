---
title: Financial Document Intelligence Agent
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

  <h1>📊 Financial Document Intelligence Agent</h1>

  <p><strong>Upload an annual report. Get structured extraction, risk classification, and an intelligence brief.</strong><br/>
  Built with LangGraph · LangFuse · RAG · RAGAS</p>

  <a href="https://huggingface.co/spaces/Matigob/financial-doc-agent">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Hugging%20Face-orange?style=for-the-badge" alt="Live Demo"/>
  </a>

  <br/><br/>

  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangGraph-Orchestration-1C7ED6?style=flat-square"/>
  <img src="https://img.shields.io/badge/LangFuse-Observability-7B2FBE?style=flat-square"/>
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-22B455?style=flat-square"/>
  <img src="https://img.shields.io/badge/RAGAS-Evaluation-E03131?style=flat-square"/>
  <img src="https://img.shields.io/badge/Docker-Deployed-2496ED?style=flat-square&logo=docker&logoColor=white"/>

</div>

<br/>

---

## The Problem

Investors, analysts, and founders process dozens of annual reports and earnings transcripts manually — copy, paste, search through PDFs. Existing tools are either too expensive (Bloomberg Terminal) or too generic (ChatGPT with no structure).

There is no accessible agent that treats a financial document as structured data: extract the KPIs, cross-reference known risk patterns, compare periods, and produce a decision-ready brief.

This system does that.

---

## What It Does

| Feature | Detail |
|---|---|
| 📄 **PDF ingestion** | Upload annual reports, earnings transcripts, or 10-K/20-F filings |
| 🔢 **Structured extraction** | Revenue, EBITDA, gross margin, cash, debt, YoY growth, guidance — via Pydantic schema |
| ⚠️ **Risk classification** | RAG over a financial red flags library — matches extracted risks against known patterns |
| 📋 **Intelligence brief** | 3-paragraph plain-English report: performance summary, risk assessment, outlook |
| 🔁 **Reflection loop** | Critique model scores brief 0–9 (GROUNDED · ACTIONABLE · CONCISE), retries if below threshold |
| 📊 **Peer comparison** | Run two documents side-by-side for multi-company analysis |

---

## Architecture

The system is a **LangGraph StateGraph** with 5 nodes and a conditional retry edge.

```
START
  │
  ▼
parse_document      ← PyMuPDF + pdfplumber → clean text + tables
  │
  ▼
extract_metrics     ← LLM (gemini-2.5-flash) → FinancialMetrics (Pydantic)
  │                   revenue · EBITDA · margins · risks · guidance · management tone
  ▼
classify_risks      ← RAG (ChromaDB red flags library)
  │                   top-5 matched risk patterns injected into report context
  ▼
generate_report     ← LLM (gemini-2.5-flash) → 3-paragraph intelligence brief
  │
  ▼
critique_report     ← LLM (gemini-2.5-flash-lite) → scores 0–9
  │          │
  │    score < 7  ──► retry generate_report (max 2×)
  │
  ▼
END
```

**Nodes:** 5 · **Conditional edges:** retry loop on critique · **State:** `DocumentState` shared across all nodes

---

## Evaluation

| Metric | Result | Method |
|---|---|---|
| RAG answer relevancy | **0.842** | RAGAS · 6 documents |
| RAG faithfulness | **0.701** | RAGAS · 5 documents |

**Cost per document (LangFuse tracked):**

| Document | Cost |
|---|---|
| Airbnb 10-K FY2024 | $0.0049 |
| Uber 10-K FY2024 | $0.0050 |
| Sea Limited 20-F FY2024 | $0.0043 |
| Grab Holdings 20-F (multi-year) | $0.0173 |
| Grab vs Sea (peer comparison) | $0.0096 |

> **Note on faithfulness:** The system generates reports from both structured extraction (exact numbers: revenue, EBITDA) and RAG-retrieved red flag patterns. RAGAS faithfulness checks only the red flags as context — exact financial figures are not in the retrieval context, so they score as "ungrounded". This reflects the dual-source architecture, not hallucination.

---

## Sample Documents

Demo reports are pre-generated from real SEC filings:

| Company | Filing | Period |
|---|---|---|
| Grab Holdings Limited | Form 20-F | FY2022 · FY2023 · FY2024 |
| Sea Limited | Form 20-F | FY2024 |
| Uber Technologies | Form 10-K | FY2024 |
| Airbnb, Inc. | Form 10-K | FY2024 |

> Demo mode: live analysis is disabled in the public deployment to avoid burning API credits. All pre-computed reports are fully interactive.

---

## Tech Stack

| Technology | Role |
|---|---|
| **LangGraph** | Agent orchestration — StateGraph, conditional edges, retry loop |
| **LangFuse** | Full observability — traces, cost per document, latency per node |
| **LangChain + ChromaDB** | RAG layer — HuggingFace embeddings, financial red flags retrieval |
| **PyMuPDF + pdfplumber** | PDF parsing — prose extraction + table extraction |
| **Pydantic** | Structured extraction schema — typed output from LLM |
| **RAGAS** | Automated evaluation — faithfulness and answer relevancy |
| **FastAPI** | Backend REST API |
| **Docker** | Containerized deployment on Hugging Face Spaces |

---

## Run Locally

```bash
git clone https://github.com/mateusz-gob1/financial-doc-agent
cd financial-doc-agent

python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS / Linux

pip install -r requirements.txt
cp .env.example .env
```

Add your API keys to `.env`:

```env
OPENROUTER_API_KEY=      # LLM calls (OpenRouter)
LANGFUSE_PUBLIC_KEY=     # observability
LANGFUSE_SECRET_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

**Run the demo dashboard:**
```bash
uvicorn api.main:app --reload
# open http://localhost:8000
```

**Run the agent on a PDF:**
```bash
python -c "
from agents.graph import build_graph
graph = build_graph()
result = graph.invoke({'pdf_path': 'data/sample_docs/Grab Holdings Limited — Form 20-F-2024.pdf'})
print(result['report'])
"
```

---

## Project Structure

```
financial-doc-agent/
├── agents/
│   ├── graph.py            # LangGraph StateGraph definition
│   ├── nodes.py            # 5 nodes: parse · extract · classify · generate · critique
│   └── state.py            # DocumentState TypedDict
├── tools/
│   ├── pdf_parser.py       # PyMuPDF + pdfplumber combined extraction
│   ├── vector_store.py     # ChromaDB red flags retrieval
│   ├── llm_client.py       # OpenRouter + LangFuse callback setup
│   ├── cost_tracker.py     # per-document cost accumulation
│   ├── history_store.py    # SQLite report history
│   └── comparator.py       # multi-period and peer comparison logic
├── schemas/
│   └── financial.py        # FinancialMetrics Pydantic model
├── evaluation/
│   └── ragas_eval.py       # RAGAS faithfulness + answer relevancy
├── api/
│   └── main.py             # FastAPI backend
├── frontend/               # vanilla JS dashboard (index.html)
├── data/
│   ├── sample_docs/        # SEC filings (Grab, Sea, Uber, Airbnb)
│   ├── red_flags/          # RAG knowledge base
│   └── demo_data.json      # pre-computed reports for demo mode
├── scripts/
│   ├── generate_demo.py    # regenerate demo_data.json
│   └── inject_demo_peer.py # inject peer comparison entry
├── vault/                  # Architecture.md · Evaluation-Log.md
└── Dockerfile
```
