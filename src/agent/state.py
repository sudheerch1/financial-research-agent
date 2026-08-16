from typing import TypedDict, Annotated
import operator


class AgentState(TypedDict):
    # --- Input ---
    question: str                          # original user question

    # --- Planning ---
    research_goals: list[str]              # 3-7 goals decomposed by node_plan

    # --- RAG ---
    rag_results: str                       # raw results from RAG search
    confidence_score: float                # 0.0–1.0 evidence quality score
    iteration: int                         # RAG retry counter (max 5)
    rewritten_queries: list[str]           # query variations from node_rewrite
    best_rag_results: str                  # highest-confidence rag_results seen across retries
    best_confidence_score: float           # the confidence_score that produced best_rag_results
    
    # --- ReAct Tool Loop ---
    tool_calls_made: Annotated[list[str], operator.add]  # accumulates tool names
    tool_results: Annotated[list[dict], operator.add]    # accumulates tool outputs

    # --- Reflection ---
    goals_met: list[bool]                  # one bool per research goal
    gaps_identified: list[str]             # goals not yet satisfied
    reflection_cycles: int                 # reflection retry counter (max 2)

    # --- Conflict Detection ---
    conflict_detected: bool
    is_critical: bool
    conflict_details: str

    # --- Human Review ---
    human_approval_decision: str           # "sec_primary" | "balanced" | "disputed"

    # --- Output ---
    report_mode: str                       # "standard" | "warning" | "disputed" | etc.
    final_report: str                      # the finished investment research report
    company: str  # detected from question: "apple", "microsoft", "amazon", or "" if undetermined
