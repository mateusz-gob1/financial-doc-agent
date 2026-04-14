# Architecture

## System Overview

Financial Document Intelligence Agent — LangGraph pipeline that processes financial PDFs (annual reports, earnings transcripts) into structured data and risk reports.

## LangGraph StateGraph

```
START
  │
  ▼
parse_document      ← PyMuPDF + pdfplumber → clean text + tables
  │
  ▼
extract_metrics     ← LLM (gemini-2.5-flash) → FinancialMetrics (Pydantic)
  │
  ▼
classify_risks      ← RAG (ChromaDB red flags library) → enriches risk_score
  │
  ▼
generate_report     ← LLM (gemini-2.5-flash) → plain English summary
  │
  ▼
critique_report     ← LLM (gemini-2.5-flash-lite) → scores 0–9
  │          │
  │    score < 7  ──→ retry generate_report (max 2×)
  │
  ▼
END
```

**Nodes:** 5  
**Conditional edges:** retry loop on critique_report  
**State:** DocumentState dataclass (shared across all nodes)

## Models

| Task | Model | Reason |
|---|---|---|
| Extraction + report | google/gemini-2.5-flash | Best quality for structured output |
| Critique | google/gemini-2.5-flash-lite | Fast, cheap — only needs scoring |

All calls via OpenRouter.

## PDF Parsing

Two-library approach:
- **PyMuPDF** (`fitz`) — fast prose extraction, handles most PDFs
- **pdfplumber** — table extraction, structured financial data in tabular form

Combined output truncated to 40k chars before LLM call.

## RAG — Red Flags Library

ChromaDB collection of financial red flags (e.g. "declining gross margins", "customer churn", "debt covenant breach risk"). Retrieved top-k chunks used in `classify_risks` node to cross-reference extracted risks against known patterns.

Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local, no API cost).

## Observability

LangFuse traces on every LLM call from day 1:
- cost per document
- latency per node
- extraction accuracy over time

## Stack

| Technology | Role |
|---|---|
| LangGraph | Orchestration — StateGraph |
| LangFuse | Observability — traces, cost, latency |
| LangChain + ChromaDB | RAG — red flags library |
| PyMuPDF + pdfplumber | PDF parsing |
| Pydantic | Structured extraction schema |
| FastAPI | Backend REST API |
| Vanilla JS | Frontend dashboard |
| Docker | Deployment |
| RAGAS | Evaluation layer |
