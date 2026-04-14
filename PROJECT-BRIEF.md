# Financial Document Intelligence Agent
> Projekt 2 — wstępny brief (2026-04-13)

---

## Problem

Inwestorzy, analitycy i founders muszą przetwarzać dziesiątki raportów rocznych,
earnings transcriptów i pitch decków. Ręcznie — kopiuj, wklej, szukaj w PDF.
Istniejące narzędzia są albo zbyt drogie (Bloomberg Terminal) albo zbyt generyczne
(ChatGPT bez struktury).

Luka: brak dostępnego agenta który traktuje dokument finansowy jak dane do analizy —
ekstrahuje KPI, flaguje ryzyka, porównuje okresy, generuje raport dla decydenta.

---

## Co system robi

1. **Upload PDF** — raport roczny, earnings transcript, pitch deck
2. **Structured extraction** — revenue, EBITDA, guidance, key risks, management commentary
3. **Risk flagging** — RAG nad bazą red flags (rosnący dług, spadające marże, odpływy klientów)
4. **Porównanie okresów** — Q4 2024 vs Q4 2023, rok do roku
5. **Plain-English summary** — raport dla kogoś bez backgroundu finansowego
6. **Risk score** — Low / Medium / High z uzasadnieniem

---

## Dlaczego ten domain

- Publiczne dane: EDGAR (SEC filings), raporty IR firm z SEA (Grab, Sea, Shopee)
- Demo natychmiast zrozumiałe dla każdego
- Bezpośrednie trafienie w Singapore i HK gdzie finance/VC jest główną branżą
- Nie wymaga domain expertise — system jest analitykiem, nie autor

---

## Architektura (wstępna)

```
START
  │
  ▼
parse_document      ── PyMuPDF → czysty tekst, tabele, metadane
  │
  ▼
extract_metrics     ── LLM → Pydantic schema (revenue, margins, guidance, risks)
  │
  ▼
classify_risks      ── RAG nad bazą financial red flags
  │
  ▼
compare_periods     ── (opcjonalnie) diff między dwoma dokumentami / okresami
  │
  ▼
generate_report     ── plain English summary + risk score
  │
  ▼
critique_report     ── model sprawdza jakość, retry jeśli < threshold
  │
  ▼
END
```

---

## Co nowego vs Football Agent

| Obszar | Football Agent | Financial Doc Agent |
|---|---|---|
| Input | API + scraping | PDF upload |
| Output | narracyjny briefing | structured extraction + risk report |
| RAG | artykuły prasowe | baza financial red flags |
| Frontend | vanilla JS dashboard | Next.js + TypeScript (nowe!) |
| Storage | JSON pliki | SQLite — historia dokumentów |
| Nowe lib | — | PyMuPDF, pdfplumber, Pydantic schemas |

---

## Stack

| Technologia | Rola | Nowe? |
|---|---|---|
| LangGraph | Orchestration | ✗ |
| LangFuse | Observability | ✗ |
| LangChain + ChromaDB | RAG nad red flags library | ✗ |
| RAGAS | Evaluation | ✗ |
| PyMuPDF / pdfplumber | PDF parsing | ✅ |
| Pydantic | Structured extraction schemas | ✅ |
| FastAPI | Backend | ✗ |
| Next.js + TypeScript | Frontend | ✅ |
| SQLite | Historia uploadów | ✅ |
| Docker | Deployment | ✗ |

---

## Metryki do CV

- Extraction accuracy (vs manually annotated sample)
- RAGAS faithfulness + answer relevancy
- Risk classification accuracy (LLM-as-judge)
- Latency per page (LangFuse)
- Koszt per dokument (LangFuse)

---

## Demo dokumenty (publiczne, gotowe do użycia)

- Grab Holdings Annual Report 2023/2024 (SEC EDGAR)
- Sea Limited Annual Report (SEC EDGAR)
- Apple / Amazon earnings transcripts (Motley Fool, Seeking Alpha)
- Przykładowy pitch deck (YC public decks)

---

## Narracja na rozmowie

*"Zamiast generycznego 'upload PDF i zadaj pytanie' — zbudowałem system który
traktuje dokument finansowy jak dane strukturyzowane. Wyciąga konkretne metryki,
porównuje z poprzednim okresem i generuje raport ryzyka. Demo działa na
prawdziwych raportach Graba i Sea Limited — firm które celuję jako pracodawców
w Singapurze."*

---

## TODO przed startem

- [ ] Zdecydować czy frontend to Next.js czy zostajemy przy vanilla JS + FastAPI
- [ ] Zebrać 5-10 sample dokumentów do testów i ewaluacji
- [ ] Zdefiniować Pydantic schema dla kluczowych metryk finansowych
- [ ] Sprawdzić CUAD dataset jako bazę dla RAG (dostępny publicznie)
