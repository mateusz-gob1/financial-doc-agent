import fitz  # PyMuPDF
import pdfplumber

# Keywords that signal a page contains financial data worth extracting.
# Ordered by priority — first match wins the section anchor.
FINANCIAL_SECTION_KEYWORDS = [
    "selected financial data",
    "selected consolidated financial data",
    "consolidated statements of operations",
    "consolidated statements of comprehensive",
    "consolidated balance sheet",
    "consolidated statements of cash flows",
    "item 5",       # 20-F: Operating and Financial Review
    "item 18",      # 20-F: Financial Statements
    "item 7",       # 10-K: Management's Discussion and Analysis
    "item 8",       # 10-K: Financial Statements
]

# How many pages to grab after each anchor page
PAGES_AFTER_ANCHOR = 25


def extract_pages_pymupdf(pdf_path: str) -> tuple[list[str], int]:
    """Return list of per-page text strings and total page count."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return pages, len(pages)


def find_financial_section_pages(pages: list[str]) -> list[int]:
    """
    Scan pages for financial section keywords.
    Returns a deduplicated sorted list of page indices to extract.

    Matches only in the first 300 chars of a page (section headers),
    not in body text where these phrases appear as cross-references.
    Limits to the first 4 anchor points to avoid extracting the whole doc.
    """
    anchors: list[tuple[int, int]] = []  # (page_index, keyword_priority)

    for i, page_text in enumerate(pages):
        # Only look at the top of the page — section headers appear there
        header_zone = page_text[:300].lower().strip()
        for priority, keyword in enumerate(FINANCIAL_SECTION_KEYWORDS):
            if keyword in header_zone:
                anchors.append((i, priority))
                break

    # Keep only the first 4 unique anchor pages (avoid over-extraction)
    seen: set[int] = set()
    selected: list[int] = []
    for page_idx, _ in sorted(anchors, key=lambda x: x[0]):
        if page_idx not in seen:
            seen.add(page_idx)
            for j in range(page_idx, min(page_idx + PAGES_AFTER_ANCHOR, len(pages))):
                selected.append(j)
        if len(seen) >= 4:
            break

    return sorted(set(selected))


def extract_financial_sections(pdf_path: str) -> tuple[str, int, list[int]]:
    """
    Smart extraction: find financial section pages, return only their text.

    Falls back to first 60 pages if no financial keywords found
    (e.g. for pitch decks or non-standard formats).

    Returns:
        text: concatenated text of relevant pages
        page_count: total pages in document
        extracted_pages: which page indices were used
    """
    pages, page_count = extract_pages_pymupdf(pdf_path)
    relevant = find_financial_section_pages(pages)

    if not relevant:
        # Fallback: take first 60 pages
        relevant = list(range(min(60, page_count)))

    text = "\n\n".join(pages[i] for i in relevant)
    return text, page_count, relevant


def parse_document(pdf_path: str, max_chars: int = 120_000) -> tuple[str, int]:
    """
    Parse a financial PDF into clean text ready for LLM extraction.

    Uses smart section detection to find financial data pages instead of
    naively truncating from the start of the document.

    Returns:
        text: relevant financial sections (capped at max_chars)
        page_count: total pages in document
    """
    text, page_count, extracted_pages = extract_financial_sections(pdf_path)

    header = f"[Extracted {len(extracted_pages)} of {page_count} pages: {extracted_pages[:5]}{'...' if len(extracted_pages) > 5 else ''}]\n\n"

    combined = header + text
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n\n[Truncated]"

    return combined, page_count
