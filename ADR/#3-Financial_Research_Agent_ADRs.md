# Architecture Decision Records — Financial Research Agent POC

**Project:** LangGraph agentic RAG system answering cross-company financial questions from SEC 10-K filings (Apple, Microsoft, Amazon — FY2025)
**Repos:** `financial-research-agent` (agent + FastAPI + Streamlit, ECS Fargate) · `rag-pipeline-financial` (RAG ingestion/query)
**AWS:** account `<ACCOUNT_ID>`, region `us-east-1`

Each record follows: Status → Context → Decision → Alternatives Considered → Consequences. Status is one of: **Accepted**, **Accepted — POC scope, revisit for production**, or **Accepted risk**.

---

## ADR-001: LangGraph for orchestration, not a single LLM prompt

**Status:** Accepted

**Context**
A single LLM call can't verify its own confidence, retry with new evidence, call external tools conditionally, or pause for human input mid-task. The system needed genuine conditional control flow driven by runtime state, not a fixed sequence of prompts.

**Decision**
Built a 9-node LangGraph state machine (`node_plan → node_rag_search → node_assess_confidence → node_react_loop → node_rewrite_queries → node_reflect → node_detect_conflict → node_human_review → node_write_report`), with two conditional routing functions (`route_after_confidence`, `route_after_reflection`) that inspect state and decide the next node at runtime.

**Alternatives Considered**
- Single large prompt with instructions to "check your work" — rejected: no mechanism to actually verify anything or retry with new input; the model just generates text once.
- LangChain's `create_react_agent` (prebuilt) — rejected for the reasoning-loop portion specifically; see ADR-003.

**Consequences**
- Real retry loops (`route_after_confidence`, `route_after_reflection`) exist, each with a shared 5-iteration circuit breaker to prevent runaway retries.
- Genuine human-in-the-loop via `interrupt()` is only possible because execution is a resumable graph, not a single completion.
- Adds real complexity: state schema design, reducer logic for accumulating fields, and routing-order bugs become possible (mitigated by confirming exact routing priority against `edges.py` directly rather than assuming).

---

## ADR-002: Raw `boto3` Bedrock calls, not LangChain, for the orchestrator

**Status:** Accepted

**Context**
LangGraph's value is the graph/state model, not LCEL chain composition. Introducing LangChain into the orchestrator would add an abstraction layer without a corresponding benefit for hand-rolled routing logic.

**Decision**
Every LLM call inside `nodes.py`/`edges.py` is a direct `bedrock-runtime.invoke_model()` call. LangChain/LCEL is confined entirely to the separate `rag-pipeline-financial` repo (retrieve → rerank → dedupe chain), where chain composition is a natural fit.

**Alternatives Considered**
- LangChain throughout, for consistency — rejected: no functional benefit inside the orchestrator, and would obscure the control flow this system's value depends on being fully legible.

**Consequences**
- A common point of confusion worth pre-empting: LangGraph does not imply LangChain everywhere in the stack. Stated explicitly in the demo narrative.
- Full visibility into every prompt and every parsing step, at the cost of writing more boilerplate than a chain abstraction would require.

---

## ADR-003: Hand-rolled ReAct loop, not LangGraph's prebuilt `create_react_agent`

**Status:** Accepted

**Context**
The tool-selection loop (`node_react_loop`) needed confidence-aware retry behavior and reducer-based accumulation of tool calls/results (`Annotated[list, operator.add]`) — control surfaces the prebuilt agent doesn't expose.

**Decision**
Wrote a custom ReAct loop: the LLM reads the question, goals, and evidence gaps, selects only tools that can close remaining gaps (never blindly calling every available tool), and results accumulate via reducers rather than overwriting.

**Alternatives Considered**
- `create_react_agent` — rejected: no access to the confidence-aware retry logic or the specific reducer pattern this system depends on; would have been a black box at exactly the point where the demo's strongest evidence of "genuine agency" lives.

**Consequences**
- Full control over tool-call deduplication policy, retry budget, and iteration limits.
- Known gap (accepted, see ADR-008): `remaining_tools` is currently a prompt hint only — no code-level enforcement stops the LLM from re-calling an already-used tool.

---

## ADR-004: DynamoDB for LangGraph checkpointing, not an RDBMS

**Status:** Accepted

**Context**
The access pattern for graph state is a single-key lookup — one JSON-shaped `AgentState` blob per `thread_id` — with no joins or cross-thread queries. The state schema has also evolved over the project's life (`best_rag_results`, `best_confidence_score` added later).

**Decision**
Used `DynamoDBSaver` as the LangGraph checkpointer, backed by a dedicated DynamoDB table with a 7-day TTL.

**Alternatives Considered**
- Postgres via `langgraph-checkpoint-postgres` — a legitimate alternative (Postgres's `JSONB` column type serves the same key-value access pattern; it isn't a mismatch of relational vs. unstructured data). Not chosen because there was no existing Postgres instance in this AWS-native, serverless-first stack — introducing one solely for checkpointing would have added a second database technology for a narrow purpose.

**Consequences**
- Native TTL handles checkpoint expiry without a scheduled cleanup job.
- Schema-less item storage absorbed `AgentState`'s growth without migrations.
- AWS-locked: this choice is not portable off AWS without a checkpointer swap. Accepted given the project's AWS-native scope.
- Confirmed learning: LangGraph writes a checkpoint after every node during a continuous run, but only round-trips to DynamoDB on resumption after `interrupt()`, crash recovery, or replay — not on every intermediate step during a live run.

---

## ADR-005: Two-layer conflict detection (LLM reasoning + deterministic verification)

**Status:** Accepted

**Context**
An LLM-only conflict check has two independent failure modes: it can reason incorrectly about whether two figures actually disagree (a reasoning error), and it can hallucinate that a figure exists in the evidence when it doesn't (a fabricated citation). A single defense layer cannot catch both.

**Decision**
`node_detect_conflict` runs two layers: (1) an LLM reasoning pass judging whether cited figures across sources genuinely disagree, and (2) `_verify_cited_figures`, a deterministic code-level check (substring matching + ~1% arithmetic tolerance) validating cited figures against the actual retrieved source text. Layer 2 is **veto-only** — it can force a claimed conflict to `False`/empty, but can never confirm or upgrade a conflict claim to `True`.

**Alternatives Considered**
- LLM reasoning alone — rejected: no way to catch a confidently-stated but fabricated citation.
- Deterministic checking alone — rejected: cannot judge semantic disagreement (e.g., different units, different fiscal-year scope) the way an LLM reasoning pass can.

**Consequences**
- Two distinct failure modes are each covered by the layer suited to catching them.
- The veto-only asymmetry is deliberate: a code check should be trusted to say "this is definitely wrong," not "this is definitely right" — an LLM's affirmative judgment is still required to raise a conflict in the first place.
- `conflict_detected AND is_critical` (an AND-gate) determines whether human review actually fires — conflict alone routes to a warning in the report rather than halting execution.

---

## ADR-006: Security-group-based isolation instead of a private subnet + NAT Gateway

**Status:** Accepted — POC scope, revisit for production

**Context**
The Fargate task and ALB both need to reach the internet (Bedrock, Tavily, yfinance) and each other. A traditional pattern places the task in a private subnet behind a NAT Gateway; this project's actual deployed topology (confirmed via `aws ecs describe-services`) uses one public subnet for both, with no NAT Gateway at all.

**Decision**
Isolation is enforced identity-based, not topology-based: the task's security group accepts inbound traffic on port 8000 only from the ALB's security group (referenced by group ID, not a CIDR range). The task has a public IP but is unreachable directly, since unsolicited inbound traffic is dropped at the security group before reaching uvicorn.

**Alternatives Considered**
- Private subnet + NAT Gateway — the conventional production pattern; provides topology-level isolation. Rejected for this POC on cost grounds: a NAT Gateway bills per-hour and per-GB for a workload at this scale, for a security guarantee already achieved via the security-group chain.

**Consequences**
- Real cost savings with an equivalent practical security outcome for a fixed, small-scale POC.
- Not the production-shaped answer at scale or under a stricter compliance posture — a private subnet remains the correct recommendation once the system needs defense-in-depth beyond a single security-group boundary.

---

## ADR-007: `run_id` async job pattern for long-running graph execution over HTTP

**Status:** Accepted

**Context**
A LangGraph run involves multiple sequential Bedrock calls and RAG round-trips — far too slow to complete inside a single blocking HTTP request/response cycle, and incompatible with `interrupt()`-based human review, which can pause indefinitely.

**Decision**
`POST /run` returns a UUID (`run_id`) immediately; the actual graph execution runs as a background task, decoupled from the HTTP connection. The client polls `GET /status/{run_id}` → `GET /result/{run_id}`, with `POST /resume/{run_id}` delivering a human decision during a paused interrupt. `run_id` is passed to LangGraph as `thread_id`, tying it directly to the DynamoDB checkpoint.

**Alternatives Considered**
- Synchronous request/response, blocking until the graph completes — rejected: incompatible with `interrupt()`'s indefinite pause, and would tie up a connection for the full multi-call duration of a run.

**Consequences**
- HTTP connections stay short and cheap; expensive work is fully decoupled from the open connection.
- Two users never collide, since `run_id`s are UUIDs and the server doesn't track "who" is asking.
- Introduces the RunStore correctness dependency documented in ADR-008.

---

## ADR-008: In-memory RunStore, deferred to DynamoDB migration (accepted risk)

**Status:** Accepted risk — must be resolved before `desiredCount > 1`

**Context**
`GET /status/{run_id}` needs somewhere to look up a run's current status between when `/run` returns and the graph finishes. The current implementation (`InMemoryRunStore`) is a plain Python dictionary inside the single running Fargate task's process.

**Decision**
Shipped with in-memory `RunStore` for the POC, with the correctness boundary explicitly known and documented rather than discovered later: this is safe only at `desiredCount: 1`. Migration path is already identified — move to DynamoDB, the same pattern already proven for LangGraph checkpointing.

**Alternatives Considered**
- Build the DynamoDB-backed RunStore immediately — deferred due to POC time constraints; the correctness boundary was judged acceptable to carry as a disclosed gap rather than block the demo timeline.

**Consequences**
- At `desiredCount: 1`, fully correct — no known failure mode.
- The moment more than one task runs behind the ALB, `GET /status` can be round-robined to a task that never saw a given `run_id`'s creation, producing sporadic 404s; a task restart also loses in-flight status. This is a hard blocker for horizontal scaling, not optional hardening.
- Explicitly disclosed in the demo narrative as a known, understood, and already-diagnosed gap — not an oversight.

---

## ADR-009: No authentication on the ALB endpoint for the POC (accepted risk)

**Status:** Accepted risk — revisit before any non-demo exposure

**Context**
`/run`, `/status`, `/resume`, `/result` currently have no auth check — anyone with the ALB's DNS name has full access. The internal RAG pipeline (API Gateway → Lambda) already enforces an API key; the main application path does not yet mirror that pattern.

**Decision**
Deferred implementing the FastAPI API-key middleware for the initial demo, given a fixed, known, small audience and time constraints. The fix is scoped and ready (shared-secret check in FastAPI middleware, same pattern as the RAG pipeline's `apiKeyRequired`), but not yet implemented as of the demo date.

**Alternatives Considered**
- API Gateway + API key in front of the ALB — has a bypass caveat: the ALB's own DNS name remains reachable directly unless separately restricted.
- Cognito + API Gateway (JWT-based identity) — the correct production answer once there's a real multi-user audience needing per-identity revocation and audit trail; judged as over-engineering for a fixed-audience demo.

**Consequences**
- Real exposure for as long as the ALB DNS name is not shared beyond the intended demo audience.
- HTTPS is a prerequisite multiplier for any of the above — an API key or JWT sent over the current HTTP-only listener would still be interceptable in transit, so this gap and the missing HTTPS listener (ADR-010) must be closed together for either to meaningfully hold.

---

## ADR-010: HTTP-only ALB listener; HTTPS deferred (accepted risk)

**Status:** Accepted risk — revisit before production

**Context**
TLS on the ALB requires an ACM certificate issued against a domain ACM can validate — the shared `*.elb.amazonaws.com` hostname cannot be certified, since it isn't uniquely owned by this deployment.

**Decision**
Shipped with the ALB's HTTP:80 listener only. HTTPS requires a custom domain (Route 53 or another registrar), an ACM-issued certificate validated via a DNS record, and a new 443 listener with the 80 listener redirected — none of which was in scope for the demo timeline.

**Alternatives Considered**
- None — this is a genuine external dependency (domain ownership), not a design trade-off with a viable near-term alternative within AWS alone.

**Consequences**
- All traffic, including any future API key or JWT, currently travels unencrypted.
- Explicitly named as the prerequisite that makes every other access-control layer (ADR-009) meaningful once implemented.

---

## ADR-011: Whole-question RAG query on first pass, not decomposed per-goal retrieval

**Status:** Accepted — POC scope, revisit for production

**Context**
`node_plan` decomposes the user's question into 3–7 explicit `research_goals`. It would be reasonable to assume those goals drive retrieval — e.g., one RAG query per goal, merged afterward. They do not: `research_goals` is consumed only by `node_reflect` (gap detection) and `node_react_loop` (tool-selection hints), never by RAG query construction.

**Decision**
`node_rag_search` sends `state["question"]` — the full, unmodified original question exactly as the user typed it — as the RAG query on the first pass. Only on a low-confidence retry does the query change, and even then it's replaced with a single variant from `node_rewrite_queries`, not a goal-derived query.

**Alternatives Considered**
- Issue one RAG query per decomposed research goal, then merge results — deferred: would require restructuring `node_rag_search` into a loop with N RAG calls plus a merge/de-dup step, adding latency for a benefit that the existing reflect → tool-call safety net already covers for typical demo-scope questions.

**Consequences**
- A compound question (e.g., asking for one company's revenue *and* a cross-company margin comparison in the same sentence) is answered from a single combined RAG call, not a call per sub-question — retrieval quality depends on the embedding model handling compound queries adequately.
- The system's compensating mechanism is `node_reflect` + `node_react_loop`: if the single RAG pass doesn't fully cover every goal, the gap is identified and can be closed via tool calls (financial data tool, Tavily) rather than a second, goal-targeted RAG pass.
- Verified against `nodes.py`: for a question combining a single-company FY2025 metric with a three-way margin comparison, the first-pass RAG call is exactly one query covering the full compound question — not three separate retrievals.

---

## ADR-012: Single-company extraction limits multi-company RAG filtering (accepted risk)

**Status:** Accepted risk — revisit if multi-company comparison questions become a demo priority

**Context**
`rag-pipeline-financial`'s query endpoint accepts a `company` filter to scope retrieval to Apple-, Microsoft-, or Amazon-specific chunks. The orchestrator derives this filter via `_extract_company()`, a keyword-matching function that checks for company names/tickers one at a time and returns on the first match.

**Decision**
`_extract_company()` returns a single string (or an empty string if none matched) — never a list — even when a question references multiple companies (e.g., "compare Apple's net sales to Microsoft and Amazon's gross margin"). The first RAG call is filtered to whichever single company is detected first in the question text; the rest are effectively unfiltered-for on that pass.

**Alternatives Considered**
- Extend `_extract_company()` to return every company mentioned, and either fan out one filtered RAG call per company or pass a multi-value filter to the RAG API — deferred: `rag-pipeline-financial`'s query endpoint would need to accept a list-based filter, and the orchestrator would need a fan-out/merge step, neither of which was built within the POC timeline.

**Consequences**
- On a genuinely multi-company question, the first-pass RAG call retrieves chunks scoped to one company only; data for the other companies named in the question must come from `node_react_loop`'s tool calls (the financial data tool covers stock/ratio-style queries reasonably well) or a later retry — not from that first RAG call.
- Verified against `nodes.py`: for a question asking about Apple's FY2025 net sales *and* a gross-margin comparison against Microsoft and Amazon, `_extract_company()` detects `"apple"` and passes `company="apple"` to the first RAG call, despite the question also requiring Microsoft and Amazon data.
- Compounds with ADR-011: a multi-company, multi-metric question is both company-filtered to one company *and* sent as one unsplit query on the first pass. Both simplifications are covered by the same downstream reflect/tool-call safety net — neither is handled at the RAG layer itself.

---

## Summary table

| ADR | Decision | Status |
|---|---|---|
| 001 | LangGraph state machine over single-prompt | Accepted |
| 002 | Raw `boto3` Bedrock calls, no LangChain in orchestrator | Accepted |
| 003 | Hand-rolled ReAct loop over prebuilt agent | Accepted |
| 004 | DynamoDB checkpointing over RDBMS | Accepted |
| 005 | Two-layer conflict detection (LLM + deterministic veto) | Accepted |
| 006 | Security-group isolation over private subnet + NAT | Accepted — POC scope |
| 007 | `run_id` async job pattern | Accepted |
| 008 | In-memory RunStore | Accepted risk |
| 009 | No auth on ALB endpoint | Accepted risk |
| 010 | HTTP-only ALB listener | Accepted risk |
| 011 | Whole-question RAG query, not per-goal decomposition | Accepted — POC scope |
| 012 | Single-company extraction limits multi-company filtering | Accepted risk |

---

*Compiled from confirmed architecture, routing logic, and infrastructure state as of the September 10, 2026 demo. ADR-011 and ADR-012 added September 12, 2026, following direct confirmation of RAG query and company-filter behavior against `nodes.py`. "Accepted risk" entries are disclosed gaps with an identified fix, not unknowns.*
