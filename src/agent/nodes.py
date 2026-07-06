import boto3
import json
from src.agent.state import AgentState
from langgraph.types import interrupt
import re

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
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": prompt}]
        })
    )
    body = json.loads(response["body"].read())
    return body["content"][0]["text"]


# ── Node 1 ──────────────────────────────────────────────────────────────────

def _extract_company(question: str) -> str:
    """Deterministically detects which company a question is about, from the
    fixed set covered by this demo (Apple, Microsoft, Amazon). Returns "" if
    none detected — callers must handle the empty case (no filter applied)."""
    q = question.lower()
    if "apple" in q or "aapl" in q:
        return "apple"
    if "microsoft" in q or "msft" in q:
        return "microsoft"
    if "amazon" in q or "amzn" in q:
        return "amazon"
    return ""


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

    company = _extract_company(state["question"])

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
        "company": company,
    }

# ── Node 2 ──────────────────────────────────────────────────────────────────
def node_rag_search(state: AgentState) -> dict:
    """Searches SEC filings via RAG pipeline API."""
    import requests
    import os

    query = state["question"]
    if state.get("rewritten_queries"):
        query = state["rewritten_queries"][0]
    
    print(f"   📤 RAG query sent: {query}")
    
    try:
        response = requests.post(
        os.environ.get("RAG_API_ENDPOINT", "https://pfjtvbl6ck.execute-api.us-east-1.amazonaws.com/prod/query"),
        headers={
            "x-api-key": os.environ.get("RAG_API_KEY", ""),
            "Content-Type": "application/json"
        },
        json={"question": query, "company": state.get("company", "")},
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
    """LLM decides which tools to use, scoped to the question's time window."""
    from src.tools.financial_tools import TOOLS

    prompt = f"""You are a financial research agent deciding which tools to use.

Research question: {state["question"]}
Research goals: {json.dumps(state["research_goals"], indent=2)}
Evidence so far: {state["rag_results"]}
Tools already called: {state["tool_calls_made"]}

STEP 1 — Determine scope: Identify the specific fiscal year(s) or date range this
question is actually asking about (e.g. "last 3 years" relative to the most recent
fiscal year in the evidence above means exactly those 3 fiscal years — no more,
no fewer). Do not include any year outside that range unless the question
explicitly asks for forward-looking guidance, projections, or "most recent" data.

STEP 2 — Available tools:
- get_stock_price(ticker) — current price, 52w high/low, volume
- get_financial_ratios(ticker) — PE, PB, debt/equity, ROE, margins
- get_earnings_history(ticker) — quarterly EPS trend
- get_company_news(ticker) — recent headlines
- get_competitor_comparison(ticker) — sector/industry/market cap
- search_web(query) — macro and news context
- check_data_completeness — always call this last

When calling search_web, the query string MUST explicitly include the fiscal year(s)
identified in Step 1 (e.g. "Apple fiscal year 2024 revenue", not just "Apple revenue").
Never write a search_web query that could plausibly return data for years outside
the identified scope.


Decide which tools to call. Extract the ticker symbol from the question.
For any search_web call, include an "in_scope_years" arg listing the fiscal year(s)
identified in Step 1 as 4-digit strings (e.g. ["2022", "2023", "2024"]).

Respond with a JSON array of tool calls:
[
  {{"tool": "get_stock_price", "args": {{"ticker": "AAPL"}}}},
  {{"tool": "search_web", "args": {{"query": "...", "in_scope_years": ["2022", "2023", "2024"]}}}},
  {{"tool": "check_data_completeness", "args": {{}}}}
]

Return ONLY the JSON array, no other text."""

    result = _call_llm(prompt)
    tool_calls = _safe_json_parse(result, expect_list=True)

    new_tool_names = []
    new_tool_results = []

    for call in tool_calls:
        tool_name = call["tool"]
        args = call.get("args", {})

        if tool_name not in TOOLS:
            print(f"   ⚠️ LLM requested unknown tool '{tool_name}' — skipped")
            continue

        tool_fn = TOOLS[tool_name]

        if tool_name == "check_data_completeness":
            tool_output = tool_fn(state)
        else:
            tool_output = tool_fn(**args)

        new_tool_names.append(tool_name)
        new_tool_results.append(tool_output)
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

    print(f"   ✏️ Rewritten queries: {queries}") 
    
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


def _extract_dollar_amounts_millions(text: str) -> list[float]:
    """Extract all $ figures from text, normalized to millions, so they
    can be compared numerically (used to check if a cited figure is a
    legitimate derived difference between two real source figures)."""
    figures = []
    for match in re.finditer(r"\$[\d,]+\.?\d*\s*(?:billion|million)?", text):
        s = match.group()
        num_match = re.search(r"[\d,]+\.?\d*", s)
        if not num_match:
            continue
        try:
            value = float(num_match.group().replace(",", ""))
        except ValueError:
            continue
        if "billion" in s.lower():
            value *= 1000
        figures.append(value)
    return figures


def _verify_cited_figures(conflict_details: str, rag_results: str, tool_results: list) -> bool:
    """Returns True if every dollar-amount figure cited in conflict_details
    either (a) appears verbatim in the source text, or (b) is the arithmetic
    difference between two figures that DO appear in the source text (e.g.
    the LLM correctly stating the size of a discrepancy between two verified
    figures) — within ~1% tolerance for rounding. Returns False only if a
    cited figure is neither present nor derivable this way, which is a
    stronger signal of actual fabrication rather than legitimate arithmetic
    on already-verified figures."""
    source_text = rag_results + " " + json.dumps(tool_results)
    source_values = _extract_dollar_amounts_millions(source_text)

    cited_figures = re.findall(r"\$[\d,]+\.?\d*\s*(?:billion|million)?", conflict_details)

    for figure in cited_figures:
        numeric_part = re.search(r"[\d,]+\.?\d*", figure)
        if not numeric_part:
            continue
        raw_str = numeric_part.group()

        # (a) literal match — original behavior preserved
        if raw_str in source_text:
            continue

        # (b) derived match — is this the difference between two real source figures?
        try:
            cited_value = float(raw_str.replace(",", ""))
        except ValueError:
            return False
        if "billion" in figure.lower():
            cited_value *= 1000

        is_derived = False
        for i in range(len(source_values)):
            for j in range(i + 1, len(source_values)):
                diff = abs(source_values[i] - source_values[j])
                if diff > 0 and abs(diff - cited_value) / diff < 0.01:
                    is_derived = True
                    break
            if is_derived:
                break

        if not is_derived:
            return False

    return True

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

IMPORTANT: A conflict must be between two different things — either two different sources
reporting different values for the same metric/period, or a source containing an actual
mathematical inconsistency (e.g. stated totals that don't sum correctly). Do NOT flag a
conflict based on your own paraphrase or summary of a single source's wording. If you believe
a single source contradicts itself, you must quote the exact two phrases verbatim from that
source's text in conflict_details that you believe contradict each other. If you cannot quote
two exact, verbatim phrases that genuinely contradict each other, do not flag a conflict —
re-read the source text carefully, since a single coherent statement (e.g. correctly describing
one period as a decline and a different period as growth) is not a contradiction.

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

    print(f"   🔎 [conflict check inputs] RAG: {state['rag_results'][:200]}")
    print(f"   🔎 [conflict check inputs] Tools: {json.dumps(state['tool_results'])[:300]}")
    
    result = _call_llm(prompt)
    parsed = _safe_json_parse(result, expect_list=False)

    conflict_detected = parsed.get("conflict_detected", False)
    is_critical = parsed.get("is_critical", False)
    conflict_details = parsed.get("conflict_details", "")

    # Guard against hallucinated figures: if the LLM cited a specific dollar
    # amount that doesn't actually appear anywhere in the source evidence,
    # treat this as a false positive rather than trusting it.
    if conflict_detected and conflict_details:
        if not _verify_cited_figures(conflict_details, state['rag_results'], state['tool_results']):
            print(f"   ⚠️ Conflict claim rejected — cited figure not found in source text: {conflict_details}")
            conflict_detected = False
            is_critical = False
            conflict_details = ""
    
    print(f"   🔎 conflict_detected={conflict_detected} | is_critical={is_critical} | details={conflict_details}")

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
        "sec_primary": (
        "Write the report treating SEC filings (the RAG/SEC results) as the sole "
        "authoritative source for any metric where it conflicts with other evidence. "
        "When RAG/SEC results and tool results (e.g. web search, third-party aggregators) "
        "report different values for the same metric and period, use ONLY the RAG/SEC "
        "figure throughout the entire report — do not cite, blend, or average in the "
        "conflicting web-sourced figure, even as a secondary data point."),
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

Be specific, cite sources, and flag any limitations.

Only cite sources that appear verbatim in the provided source list above.
Do not name a publication, platform, or outlet unless its URL is present
in the evidence you were given for this query."""

    report = _call_llm(prompt)

    return {
        "final_report": report,
        "report_mode": mode,
    }
