from langgraph.graph import StateGraph, END

from agents.state import DocumentState
from agents.nodes import (
    node_parse_document,
    node_extract_metrics,
    node_classify_risks,
    node_generate_report,
    node_critique_report,
    should_retry,
)


def build_graph() -> StateGraph:
    graph = StateGraph(DocumentState)

    graph.add_node("parse_document", node_parse_document)
    graph.add_node("extract_metrics", node_extract_metrics)
    graph.add_node("classify_risks", node_classify_risks)
    graph.add_node("generate_report", node_generate_report)
    graph.add_node("critique_report", node_critique_report)

    graph.set_entry_point("parse_document")
    graph.add_edge("parse_document", "extract_metrics")
    graph.add_edge("extract_metrics", "classify_risks")
    graph.add_edge("classify_risks", "generate_report")
    graph.add_edge("generate_report", "critique_report")
    graph.add_conditional_edges(
        "critique_report",
        should_retry,
        {"retry": "increment_retry", "done": END},
    )
    graph.add_node("increment_retry", lambda s: {"retry_count": s.get("retry_count", 0) + 1})
    graph.add_edge("increment_retry", "generate_report")

    return graph.compile()
