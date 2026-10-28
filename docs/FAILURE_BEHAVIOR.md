# Failure Behavior & Resilience Reference

Every critical node in the LangGraph execution pipeline is governed by explicit configuration flags (`FAIL_MODE_*`) determining system resilience when external providers (Groq, Pinecone, Tavily) experience timeouts, network partitions, or rate limits.

All fail events are captured with structured metadata in the distributed telemetry ring buffer.

## Fail Mode Matrix

| Node | Configuration Flag | Default Mode | Behavior on Error / Timeout | Design Rationale |
| :--- | :--- | :---: | :--- | :--- |
| **Scope Classifier** (`classify_scope`) | `FAIL_MODE_SCOPE_CLASSIFIER` | `closed` | Fails closed: terminates the turn immediately and routes to `answer_insufficient` with a polite try-again prompt and escalation details. Setting to `open` treats errors as `in_scope`. | **Security**: Unclassified prompts must not reach vector search or LLM generation where injections or off-scope exploits could execute. |
| **Router** (`route_question`) | `FAIL_MODE_ROUTER` | `open` | Fails open: routes queries to `kb` (retrieval path) rather than dropping them. Setting to `closed` treats errors as `direct`. | **Helpfulness**: Legitimate customer questions should have the opportunity to retrieve verified policy documentation rather than receiving an unsubstantiated generic reply. |
| **Evidence Grader** (`grade_kb_evidence`, `grade_web_evidence`) | `FAIL_MODE_GRADER` | `closed` | Fails closed: treats evidence as `weak` upon LLM parser or network failure. If web search is disabled, gracefully stops at `answer_insufficient`. | **Faithfulness**: Low-confidence or unverified chunks must never be fed into generation, preventing grounded hallucinations. |
| **Faithfulness Check** (`check_claim_support`) | `FAIL_MODE_FAITHFULNESS` | `adaptive` | Adaptive policy: fails closed (`supported=False`) for policy, warranty, and billing intents to prevent corporate liability; fails open (`supported=True`) for generic inquiries. On detected ungrounded claims, executes a strict 1-turn grounded regeneration retry before human escalation. | **Balance**: Protects business liability on legal/financial policies while avoiding unnecessary user-facing errors on conversational responses. |

## Provider Outage Handling

- **Groq API HTTP 429 / 5xx:** Caught by `llm_max_retries` with exponential backoff. If exhausted, nodes transition to configured fail-mode states and log telemetry spans.
- **Pinecone Connection Timeout:** Falls back to empty retrieval candidate list; evidence grader evaluates as `weak`, routing to `answer_insufficient` without crashing the application.
- **Tavily Web Search Timeout:** Protected by `TAVILY_TIMEOUT` (15.0s) and concurrency semaphore (`TAVILY_MAX_CONCURRENCY = 5`). If Tavily times out or search fails, execution routes to `answer_insufficient`.
- **Reranker Failure:** If the cross-encoder fails to load or execute, the retrieval pipeline gracefully degrades to Pinecone cosine similarity ranking with `score_type: cosine`.
