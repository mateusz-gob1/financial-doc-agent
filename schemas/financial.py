from pydantic import BaseModel, Field
from typing import Literal


class FinancialMetrics(BaseModel):
    """Structured financial data extracted from a document by LLM."""

    company_name: str = Field(description="Company name as stated in the document")
    period: str = Field(description='Reporting period, e.g. "FY2023" or "Q4 2024"')
    document_type: Literal["annual_report", "earnings_transcript", "pitch_deck", "other"] = Field(
        description="Type of financial document"
    )

    # Income statement
    revenue_usd_m: float | None = Field(None, description="Total revenue in USD millions")
    gross_margin_pct: float | None = Field(None, description="Gross margin as a percentage, e.g. 42.5")
    ebitda_usd_m: float | None = Field(None, description="EBITDA in USD millions")
    net_income_usd_m: float | None = Field(None, description="Net income in USD millions (negative if loss)")

    # Balance sheet
    cash_usd_m: float | None = Field(None, description="Cash and equivalents in USD millions")
    debt_usd_m: float | None = Field(None, description="Total debt in USD millions")

    # Growth
    yoy_revenue_growth_pct: float | None = Field(
        None, description="Year-over-year revenue growth as a percentage"
    )

    # Qualitative
    guidance_summary: str | None = Field(
        None, description="Forward-looking statements and management guidance, 1-2 sentences"
    )
    key_risks: list[str] = Field(
        default_factory=list,
        description="Top risks identified in the document, max 5 bullet points",
    )
    management_tone: Literal["positive", "cautious", "negative"] = Field(
        description="Overall tone of management commentary"
    )

    # Risk assessment
    risk_score: Literal["Low", "Medium", "High"] = Field(
        description="Overall risk assessment of the company"
    )
    risk_justification: str = Field(
        description="2-3 sentence explanation of the risk score"
    )
