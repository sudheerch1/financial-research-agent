import boto3
import json
from src.agent.state import AgentState

# Bedrock client
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
LLM_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


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
    # Strip markdown code fences if LLM wraps response
    clean = result.strip().replace("```json", "").replace("```", "").strip()
    goals = json.loads(clean)

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

    query = state["question"]

    # If rewritten queries exist, use the first one
    if state.get("rewritten_queries"):
        query = state["rewritten_queries"][0]

    # --- STUB: replace with real API call when OpenSearch is live ---
    # response = requests.post(
    #     "https://k0injz47nk.execute-api.us-east-1.amazonaws.com/prod/query",
    #     headers={"x-api-key": "YOUR_KEY", "Content-Type": "application/json"},
    #     json={"question": query}
    # )
    # rag_results = response.json()["answer"]

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
    score = float(result.strip())

    # Stub always gets low confidence so we can test the rewrite path
    # Remove this override once RAG is live
    if "[STUB]" in state["rag_results"]:
        score = 0.9  # force proceed during development

    return {"confidence_score": score}


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
    # Strip markdown code fences if LLM wraps response
    clean = result.strip().replace("```json", "").replace("```", "").strip()
    queries = json.loads(clean)

    return {
        "rewritten_queries": queries,
        "retrieval_strategy": "B",
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
    # Strip markdown code fences if LLM wraps response
    clean = result.strip().replace("```json", "").replace("```", "").strip()
    parsed = json.loads(clean)

    return {
        "goals_met": parsed["goals_met"],
        "gaps_identified": parsed.get("gaps", []),
        "reflection_cycles": state.get("reflection_cycles", 0) + 1,
    }


# ── Node 6 ──────────────────────────────────────────────────────────────────

def node_detect_conflict(state: AgentState) -> dict:
    """Cross-checks findings for contradictions between sources."""
    prompt = f"""You are a financial data auditor checking for contradictions.

Research question: {state["question"]}
RAG results: {state["rag_results"]}
Tool results: {json.dumps(state["tool_results"], indent=2)}

Check for contradictions between data sources (e.g. revenue figures that don't match,
conflicting dates, inconsistent financial metrics).

Respond with a JSON object:
{{
  "conflict_detected": true/false,
  "is_critical": true/false,
  "conflict_details": "description of conflict or empty string"
}}

is_critical = true if the conflict would materially affect an investment decision.
Return ONLY the JSON object, no other text."""

    result = _call_llm(prompt)
    # Strip markdown code fences if LLM wraps response
    clean = result.strip().replace("```json", "").replace("```", "").strip()
    parsed = json.loads(clean)

# Force no conflict during stub development
    if "[STUB]" in state["rag_results"]:
        return {
            "conflict_detected": False,
            "is_critical": False,
            "conflict_details": "",
        }

    return {
        "conflict_detected": parsed["conflict_detected"],
        "is_critical": parsed["is_critical"],
        "conflict_details": parsed["conflict_details"],
    }


# ── Node 7 ──────────────────────────────────────────────────────────────────

def node_human_review(state: AgentState) -> dict:
    """Pauses for human decision on critical conflicts (interrupt point)."""
    # LangGraph interrupt is wired in graph.py
    # This node just surfaces the conflict for the human
    print("\n⚠️  HUMAN REVIEW REQUIRED")
    print(f"Conflict detected: {state['conflict_details']}")
    print("Options: 'sec_primary' | 'balanced' | 'disputed'")

    # In graph.py this node will be wrapped with interrupt()
    # human_approval_decision will be injected via graph.update_state()
    return {}


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