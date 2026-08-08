from src.agent.state import AgentState

def route_after_confidence(state: AgentState) -> str:
    if state["iteration"] >= 5:
        return "react_loop"
    if state["confidence_score"] >= 0.75:
        return "react_loop"
    if state.get("rewritten_queries"):
        return "retry_search"      # NEW — already have a batch, just cycle through it
    return "rewrite_queries"       # only fires once, before any rewrite exists

def route_after_reflection(state: AgentState) -> str:
    """Edge 2: after node_reflect"""
    if state["reflection_cycles"] >= 2:
        return "detect_conflict"
    if all(state["goals_met"]):
        return "detect_conflict"
    if state.get("confidence_score", 0) >= 0.75:
        return "react_loop"
    return "retry_search"

def route_after_conflict(state: AgentState) -> str:
    """Edge 3: after node_detect_conflict"""
    if state["conflict_detected"] and state["is_critical"]:
        return "human_review"
    if state["conflict_detected"]:
        return "write_report_warning"
    return "write_report_standard"


def route_after_human(state: AgentState) -> str:
    """Edge 4: after node_human_review"""
    decision = state["human_approval_decision"]
    if decision == "sec_primary":
        return "write_report_sec"
    if decision == "balanced":
        return "write_report_balanced"
    return "write_report_disputed"