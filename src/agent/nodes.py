import boto3
import json
from src.agent.state import AgentState
from langgraph.types import interrupt

# Bedrock client
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
LLM_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def _safe_json_parse(raw_text: str, expect_list: bool = False):
    """Shared helper — strips markdown fences and safely parses LLM JSON output.
    Returns [] / {} on failure instead of raising, so a malformed LLM response
    degrades the run instead of crashing it."""
    clean = raw_text.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(clean)
        if expect_list and not isinstance(parsed, list):
            raise ValueError("Expected a JSON array")
        return parsed
    except (json.JSONDecodeError, ValueError) as e:
        print(f"   ⚠️ JSON parse failed: {e} | raw: {clean[:200]}")
        return [] if expect_list else {}


def _call_llm(prompt: str) -> str:
    """Shared helper — calls Claude Haiku 4.5 via Bedrock."""
    response = bedrock.invoke_model(
        modelId=LLM_MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}]
        })
    )
    body = json.loads(response["body"].read())
    return body["content"][0]["text"]


# ── Node 1 ──────────────────────────────────────────────────────────────────

def node_plan(state: AgentState) -> dict:
    """Decomposes the question into 3-7 explicit research goals."""
    prompt = f"""You are a financial research planner.

Given this research question:
{state["question"]}

Decompose it into 3-7 specific research goals. Each goal should be a clear, 
measurable statement of what must be true before the question is fully answered.

Respond with a JSON array of strings. Example:
["Identify revenue trend over last 3 years", "Assess debt-to-equity ratio", ...]

Return ONLY the JSON array, no other text."""

    result = _call_llm(prompt)
    goals = _safe_json_parse(result, expect_list=True)

    # If planning itself failed to parse, fall back to a single generic goal
    # rather than an empty goal list (which would make every downstream
    # reflection trivially "complete").
    if not goals:
        goals = [f"Answer the research question: {state['question']}"]

    return {
        "research_goals": goals,
        "goals_met": [False] * len(goals),
        "iteration": 0,
        "reflection_cycles": 0,
        "tool_calls_made": [],
        "tool_results": [],
        "conflict_detected": False,
        "is_critical": False,
        "conflict_details": "",
        "human_approval_decision": "",
    }


# ── Node 2 ──────────────────────────────────────────────────────────────────
def node_rag_search(state: AgentState) -> dict:
    """Searches SEC filings via RAG pipeline API."""
    import requests
    import os

    query = state["question"]
    if state.get("rewritten_queries"):
        query = state["rewritten_queries"][0]

    try:
        response = requests.post(
            "https://k0injz47nk.execute-api.us-east-1.amazonaws.com/prod/query",
            headers={
                "x-api-key": os.environ.get("RAG_API_KEY", ""),
                "Content-Type": "application/json"
            },
            json={"question": query},
            timeout=30
        )
        response.raise_for_status()
        rag_results = response.json()["answer"]
    except Exception as e:
        print(f"   ⚠️ RAG API call failed: {e}")
        rag_results = f"[STUB] RAG results for: {query}"

    return {
        "rag_results": rag_results,
        "iteration": state.get("iteration", 0) + 1,
    }


# ── Node 3 ──────────────────────────────────────────────────────────────────

def node_assess_confidence(state: AgentState) -> dict:
    """Scores evidence quality 0.0–1.0."""
    prompt = f"""You are evaluating the quality of research evidence.

Research question: {state["question"]}
Evidence gathered: {state["rag_results"]}

Score the evidence quality from 0.0 to 1.0 where:
- 0.0 = completely irrelevant or empty
- 0.5 = partially relevant, missing key information  
- 1.0 = comprehensive, directly answers the research question

Respond with ONLY a number between 0.0 and 1.0. No other text."""

    result = _call_llm(prompt)
    try:
        score = float(result.strip())
    except ValueError:
        print(f"   ⚠️ Could not parse confidence score, raw: {result[:100]}")
        score = 0.5  # neutral fallback — won't falsely trigger high- or low-confidence routing

    return {"confidence_score": score}


# ── Node 3b — ReAct Tool Loop ────────────────────────────────────────────────

def node_react_loop(state: AgentState) -> dict:
    """LLM decides which tools to call based on research goals."""
    from src.tools.financial_tools import TOOLS

    # Ask LLM which tools to call
    prompt = f"""You are a financial research agent deciding which tools to use.

Research question: {state["question"]}
Research goals: {json.dumps(state["research_goals"], indent=2)}
Evidence so far: {state["rag_results"]}
Tools already called: {state["tool_calls_made"]}

Available tools:
- get_stock_price(ticker) — current price, 52w high/low, volume
- get_financial_ratios(ticker) — PE, PB, debt/equity, ROE, margins
- get_earnings_history(ticker) — quarterly EPS trend
- get_company_news(ticker) — recent headlines
- get_competitor_comparison(ticker) — sector/industry/market cap
- search_web(query) — macro and news context
- check_data_completeness — always call this last

Decide which tools to call. Extract the ticker symbol from the question.
Respond with a JSON array of tool calls:
[
  {{"tool": "get_stock_price", "args": {{"ticker": "AAPL"}}}},
  {{"tool": "check_data_completeness", "args": {{}}}}
]

Return ONLY the JSON array, no other text."""

    result = _call_llm(prompt)
    tool_calls = _safe_json_parse(result, expect_list=True)

    # Execute each tool
    new_tool_names = []
    new_tool_results = []

    for call in tool_calls:
        tool_name = call["tool"]
        args = call.get("args", {})

        if tool_name not in TOOLS:
            print(f"   ⚠️ LLM requested unknown tool '{tool_name}' — skipped")
            continue

        tool_fn = TOOLS[tool_name]

        # check_data_completeness takes state, others take kwargs
        if tool_name == "check_data_completeness":
            result = tool_fn(state)
        else:
            result = tool_fn(**args)

        new_tool_names.append(tool_name)
        new_tool_results.append(result)
        print(f"   🔧 {tool_name}({args}) → done")

    return {
        "tool_calls_made": new_tool_names,
        "tool_results": new_tool_results,
    }


# ── Node 4 ──────────────────────────────────────────────────────────────────

def node_rewrite_queries(state: AgentState) -> dict:
    """Generates 5 query variations to improve RAG retrieval."""
    prompt = f"""You are improving a financial research query to get better results.

Original question: {state["question"]}
Current evidence quality score: {state["confidence_score"]}

Generate 5 different query variations that might retrieve better evidence.
Focus on: different terminology, more specific sub-questions, alternative phrasings.

Respond with a JSON array of 5 strings.
Return ONLY the JSON array, no other text."""

    result = _call_llm(prompt)
    queries = _safe_json_parse(result, expect_list=True)

    # If rewriting fails to parse, fall back to retrying the original question
    # rather than sending an empty query list into node_rag_search.
    if not queries:
        queries = [state["question"]]

    return {
        "rewritten_queries": queries,
    }


# ── Node 5 ──────────────────────────────────────────────────────────────────

def node_reflect(state: AgentState) -> dict:
    """Evaluates which research goals are satisfied, identifies gaps."""
    prompt = f"""You are evaluating research completeness.

Research goals:
{json.dumps(state["research_goals"], indent=2)}

Evidence gathered so far:
RAG results: {state["rag_results"]}
Tool results: {json.dumps(state["tool_results"], indent=2)}

For each research goal, determine if the evidence satisfies it.
Respond with a JSON object:
{{
  "goals_met": [true/false for each goal in order],
  "gaps": ["description of each unmet goal"]
}}

Return ONLY the JSON object, no other text."""

    result = _call_llm(prompt)
    parsed = _safe_json_parse(result, expect_list=False)

    # If parsing fails, parsed is {} — fall back to the previous goals_met
    # state (or all-False if none exists yet) instead of a KeyError.
    fallback_goals_met = state.get("goals_met") or [False] * len(state["research_goals"])
    goals_met = parsed.get("goals_met", fallback_goals_met)
    gaps_identified = parsed.get("gaps", [])

    return {
        "goals_met": goals_met,
        "gaps_identified": gaps_identified,
        "reflection_cycles": state.get("reflection_cycles", 0) + 1,
    }


# ── Node 6 ──────────────────────────────────────────────────────────────────

def node_detect_conflict(state: AgentState) -> dict:
    """Cross-checks findings for contradictions between sources."""
    prompt = f"""You are a financial data auditor checking for contradictions.

Research question: {state["question"]}
RAG results: {state["rag_results"]}
Tool results: {json.dumps(state["tool_results"], indent=2)}

IMPORTANT: Different financial metrics are not directly comparable. Revenue (total net sales)
and net income (profit after expenses) are distinct figures — net income is typically much
smaller than revenue and should never be compared against it as if they represent the same thing.
Only flag a conflict when the SAME metric (e.g. revenue vs. revenue, or net income vs. net income)
shows contradictory values across sources.

IMPORTANT: Some tool results include a "data_note" field indicating they only cover the most
recent period available (not necessarily the historical period the question asks about). If a
tool result's time period doesn't overlap with the period the question or RAG results are
discussing, that is a DATA GAP, not a contradiction — do not flag it as a conflict.

Check for contradictions between data sources (e.g. revenue figures that don't match,
conflicting dates, inconsistent financial metrics of the same type) ONLY when comparing data
from the same or overlapping time periods.

Respond with a JSON object:
{{
  "conflict_detected": true/false,
  "is_critical": true/false,
  "conflict_details": "description of conflict or empty string"
}}

is_critical = true if the conflict would materially affect an investment decision.
Return ONLY the JSON object, no other text."""

    result = _call_llm(prompt)
    parsed = _safe_json_parse(result, expect_list=False)

    conflict_detected = parsed.get("conflict_detected", False)
    is_critical = parsed.get("is_critical", False)
    conflict_details = parsed.get("conflict_details", "")

    return {
        "conflict_detected": conflict_detected,
        "is_critical": is_critical,
        "conflict_details": conflict_details,
        "report_mode": "warning" if conflict_detected else "standard",
    }

# ── Node 7 ──────────────────────────────────────────────────────────────────

def node_human_review(state: AgentState) -> dict:
    decision = interrupt({
        "conflict_details": state["conflict_details"],
        "question": state["question"],
        "options": ["sec_primary", "balanced", "disputed"],
    })
    return {
        "human_approval_decision": decision,
        "report_mode": decision,  # decision is already "sec_primary" / "balanced" / "disputed" — matches mode_instruction keys directly
    }


# ── Node 8 ──────────────────────────────────────────────────────────────────

def node_write_report(state: AgentState) -> dict:
    """Produces the final investment research report."""
    mode = state.get("report_mode", "standard")

    mode_instruction = {
        "standard": "Write a confident investment research report.",
        "warning": "Write the report but include a data conflict warning section.",
        "sec_primary": "Write the report treating SEC filings as the authoritative source.",
        "balanced": "Write the report presenting both data sources with equal weight.",
        "disputed": "Write the report flagging the key metrics as disputed.",
    }.get(mode, "Write a confident investment research report.")

    prompt = f"""You are a senior financial research analyst.

{mode_instruction}

Research question: {state["question"]}

Research goals addressed:
{json.dumps(state["research_goals"], indent=2)}

Evidence:
RAG/SEC results: {state["rag_results"]}
Tool results: {json.dumps(state["tool_results"], indent=2)}

Write a structured investment research report with:
1. Executive Summary
2. Key Findings (one per research goal)
3. Data Sources
4. Confidence Assessment
5. Recommendation

Be specific, cite sources, and flag any limitations."""

    report = _call_llm(prompt)

    return {
        "final_report": report,
        "report_mode": mode,
    }