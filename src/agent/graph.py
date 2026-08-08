from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph_checkpoint_aws import DynamoDBSaver

from src.agent.state import AgentState
from src.agent.nodes import (
    node_plan,
    node_rag_search,
    node_assess_confidence,
    node_rewrite_queries,
    node_reflect,
    node_react_loop,
    node_detect_conflict,
    node_human_review,
    node_write_report,
)
from src.agent.edges import (
    route_after_confidence,
    route_after_reflection,
    route_after_conflict,
    route_after_human,
)


def build_graph():
    # ── Initialize ───────────────────────────────────────────────────────────
    builder = StateGraph(AgentState)

    # ── Add Nodes ────────────────────────────────────────────────────────────
    builder.add_node("node_plan", node_plan)
    builder.add_node("node_rag_search", node_rag_search)
    builder.add_node("node_assess_confidence", node_assess_confidence)
    builder.add_node("node_rewrite_queries", node_rewrite_queries)
    builder.add_node("node_react_loop", node_react_loop)
    builder.add_node("node_reflect", node_reflect)
    builder.add_node("node_detect_conflict", node_detect_conflict)
    builder.add_node("node_human_review", node_human_review)
    builder.add_node("node_write_report", node_write_report)

    # ── Entry Point ──────────────────────────────────────────────────────────
    builder.add_edge(START, "node_plan")

# ── Fixed Edges ──────────────────────────────────────────────────────────
    builder.add_edge("node_plan", "node_rag_search")
    builder.add_edge("node_rag_search", "node_assess_confidence")
    builder.add_edge("node_rewrite_queries", "node_rag_search")
    builder.add_edge("node_react_loop", "node_reflect")

    # ── Conditional Edge 1 — after confidence assessment ─────────────────────
    builder.add_conditional_edges(
        "node_assess_confidence",
        route_after_confidence,
        {
            "react_loop": "node_react_loop",
            "rewrite_queries": "node_rewrite_queries",
            "retry_search": "node_rag_search",   # ← new
        }
    )

    # ── Conditional Edge 2 — after reflection ────────────────────────────────
    builder.add_conditional_edges(
        "node_reflect",
        route_after_reflection,
        {
            "detect_conflict": "node_detect_conflict",
            "retry_search": "node_rag_search",
            "react_loop": "node_react_loop",   # ← new
        }
    )
    # ── Conditional Edge 3 — after conflict detection ────────────────────────
    builder.add_conditional_edges(
        "node_detect_conflict",
        route_after_conflict,
        {
            "human_review": "node_human_review",
            "write_report_warning": "node_write_report",
            "write_report_standard": "node_write_report",
        }
    )

    # ── Conditional Edge 4 — after human review ──────────────────────────────
    builder.add_conditional_edges(
        "node_human_review",
        route_after_human,
        {
            "write_report_sec": "node_write_report",
            "write_report_balanced": "node_write_report",
            "write_report_disputed": "node_write_report",
        }
    )

    # ── Exit ─────────────────────────────────────────────────────────────────
    builder.add_edge("node_write_report", END)

    # ── Compile with memory checkpointer ─────────────────────────────────────
    memory = DynamoDBSaver(
        table_name="financial-research-agent-checkpoints",
        region_name="us-east-1",
        ttl_seconds=86400 * 7,  # auto-expire after 7 days — keeps demo costs bounded
    )
    graph = builder.compile(checkpointer=memory)

    return graph


# Singleton — import this in other files
agent_graph = build_graph()