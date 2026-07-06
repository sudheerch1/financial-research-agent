"""
test_conflict_seed.py

Seeds a synthetic revenue conflict for Microsoft FY2024 by monkeypatching
the search_web tool, runs the full agent graph, and observes whether
node_human_review's interrupt() fires as expected.

Usage:
    python test_conflict_seed.py

Requires: run from the financial-research-agent repo root (so that
`from src.agent.graph import agent_graph` and `from src.tools.financial_tools
import TOOLS` resolve correctly), with your .venv activated and AWS
credentials configured (profile rag-dev or equivalent env vars set).
"""

import uuid
from langgraph.types import Command

from src.agent.graph import agent_graph
from src.tools import financial_tools


# ── Step 1: Define the conflicting stub ─────────────────────────────────────
# Real RAG-confirmed baseline (captured this session):
#   Microsoft FY2024 Total revenue = $245,122 million (~$245.1 billion)
#
# This stub reports a materially different figure for the SAME metric
# ("total revenue") and SAME period (FY2024, year ended June 30, 2024),
# with no data_note field — so node_detect_conflict cannot dismiss it as
# a metric mismatch or a data-gap. The ~$46.7B gap is intended to read as
# material enough to trip is_critical=True.

CONFLICTING_STUB = (
    "According to a widely-cited market analysis report, Microsoft's total "
    "revenue for fiscal year 2024 (year ended June 30, 2024) was $198,400 "
    "million (approximately $198.4 billion), reflecting slower cloud growth "
    "than previously projected by the company."
)


def fake_search_web(query: str, in_scope_years=None):
    """Drop-in replacement for TOOLS['search_web'] during this test run.

    Matches the REAL search_web()'s return shape exactly:
        {"tool": ..., "query": ..., "result": ..., "data_note": ...}
    (the real implementation uses "result" and "data_note" keys, not
    "source"/"content" — matching this matters because node_detect_conflict
    receives the full tool_results list as raw text, and a data_note
    claiming a period mismatch could cause the LLM to read this as a
    DATA GAP instead of a genuine conflict. We deliberately confirm the
    period aligns with FY2024 instead of using the real tool's generic
    "may not align with the fiscal period" caveat.)
    """
    print(f"   🧪 [STUB] fake_search_web called with query={query!r}")
    return {
        "tool": "search_web",
        "query": query,
        "result": CONFLICTING_STUB,
        "data_note": "Figure reflects fiscal year 2024 (year ended June 30, 2024), matching the period under discussion.",
    }


def apply_monkeypatch():
    """Patches the TOOLS registry in-place so node_react_loop picks up
    the stub instead of the real Tavily-backed search_web."""
    original = financial_tools.TOOLS.get("search_web")
    financial_tools.TOOLS["search_web"] = fake_search_web
    return original


def restore_monkeypatch(original):
    if original is not None:
        financial_tools.TOOLS["search_web"] = original


# ── Step 2: Run the graph and watch for the interrupt ───────────────────────

def run_test():
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    question = "What was Microsoft's revenue for fiscal year 2024?"

    print(f"\n{'='*70}")
    print(f"Starting run — thread_id={thread_id}")
    print(f"Question: {question}")
    print(f"{'='*70}\n")

    original_tool = apply_monkeypatch()
    try:
        result = agent_graph.invoke({"question": question}, config=config)
    finally:
        restore_monkeypatch(original_tool)

    print(f"\n{'='*70}")
    print("Graph returned (first invoke):")
    print(f"{'='*70}")

    # If the graph hit interrupt(), LangGraph surfaces it via the
    # "__interrupt__" key in the returned state rather than raising.
    if "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"]
        print("✅ INTERRUPT FIRED — node_human_review is paused.")
        print(f"Interrupt payload: {interrupt_payload}\n")
        return thread_id, config
    else:
        print("⚠️ No interrupt fired. Current state:")
        print(f"  conflict_detected = {result.get('conflict_detected')}")
        print(f"  is_critical       = {result.get('is_critical')}")
        print(f"  conflict_details  = {result.get('conflict_details')}")
        print(f"  report_mode       = {result.get('report_mode')}")
        print("\nIf conflict_detected=True but is_critical=False, the graph")
        print("routed to write_report_warning instead of human_review — the")
        print("conflict may need to be more clearly 'material' in the stub text.")
        print("\nIf conflict_detected=False entirely, check whether the LLM")
        print("saw both figures as comparable (same metric label, same period,")
        print("both figures literally present in the source text passed to")
        print("_verify_cited_figures).")
        return None, None


def resume_with_decision(config, decision: str):
    """Resumes a paused run with one of: 'sec_primary', 'balanced', 'disputed'."""
    print(f"\n{'='*70}")
    print(f"Resuming thread with decision = {decision!r}")
    print(f"{'='*70}\n")

    result = agent_graph.invoke(Command(resume=decision), config=config)

    print(f"report_mode returned: {result.get('report_mode')}")
    print(f"\n--- final_report (first 800 chars) ---")
    print(result.get("final_report", "<no report>")[:800])
    print("...\n")
    return result


if __name__ == "__main__":
    thread_id, config = run_test()

    if config is not None:
        print("\nInterrupt confirmed. To test a resume branch, re-run this")
        print("script interactively (e.g. in a Python REPL or Jupyter) so you")
        print("can call resume_with_decision(config, 'sec_primary') /")
        print("'balanced' / 'disputed' against this SAME thread_id, since")
        print("each resume consumes the paused checkpoint.")
        print(f"\nthread_id for manual resume: {thread_id}")

        # Uncomment ONE of the following to test a specific branch directly
        # in this same run (note: only works once per interrupt — the
        # checkpoint is consumed on resume):

        # resume_with_decision(config, "sec_primary")
        # resume_with_decision(config, "balanced")
        # resume_with_decision(config, "disputed")