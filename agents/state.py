from typing import TypedDict
from schemas.financial import FinancialMetrics


class DocumentState(TypedDict):
    # Input
    pdf_path: str
    document_text: str
    page_count: int

    # Extraction
    metrics: FinancialMetrics | None

    # RAG
    red_flags_context: list[str]

    # Report
    report: str
    critique_score: int
    critique_feedback: str
    retry_count: int

    # Cost tracking
    cost_usd: float

    # Error
    error: str
