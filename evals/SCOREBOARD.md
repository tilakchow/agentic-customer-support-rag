# Agentic Support RAG Improvement Scoreboard

This document tracks all quality, latency, safety, and cost metrics across baseline and sequential improvements.

---

## 1. Primary Scoreboard

| Metric | Target | Baseline (Phase 0) | Current (M0 - Fix Measurement) | Delta vs Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Recall@4 (Dev split, reranker=true)** | $\ge 0.95$ | **0.939** (95% CI: [0.890, 0.988], $n=82$) | **0.951** (95% CI: [0.902, 0.988], $n=82$) | **+0.012** |
| **Recall@4 (Dev split, reranker=false)** | - | **0.939** (95% CI: [0.890, 0.988], $n=82$) | **0.939** (95% CI: [0.890, 0.988], $n=82$) | **0.000** |
| **Precision@4 (Dev split, reranker=true)** | $\ge 0.85$ | **0.811** (95% CI: [0.744, 0.875], $n=82$) | **0.878** (95% CI: [0.820, 0.927], $n=82$) | **+0.067** |
| **Precision@4 (Dev split, reranker=false)**| - | **0.811** (95% CI: [0.744, 0.875], $n=82$) | **0.811** (95% CI: [0.744, 0.875], $n=82$) | **0.000** |
| **MRR (Dev split, reranker=true)** | $\ge 0.90$ | **0.886** (95% CI: [0.824, 0.945], $n=82$) | **0.937** (95% CI: [0.882, 0.980], $n=82$) | **+0.051** |
| **MRR (Dev split, reranker=false)** | - | **0.886** (95% CI: [0.824, 0.945], $n=82$) | **0.886** (95% CI: [0.824, 0.945], $n=82$) | **0.000** |
| **Top Score In-KB (p10, reranker=true)** | - | **0.490** (cosine, $n=82$) | **0.179** ($n=82$, in-KB only $n=71$: **0.641**) | - |
| **Top Score Out-of-Scope (p90, reranker=true)**| - | **0.393** (cosine, $n=24$) | **0.001** ($n=24$, max: **0.0057**) | - |
| **Score Gap (p10 In-KB - p90 OOS)** | $> 0.15$ | **+0.097** (separates at ~0.44) | **+0.178** ($n=82$), **+0.640** (in-KB only $n=71$) | **+0.081** |
| **Grader In-KB Bypass Rate (thresh=0.45)** | $\ge 0.90$ | **0.000** (threshold=0.85) | **0.986** (70/71 bypassed, 95% CI: [0.958, 1.000], $n=71$) | **+0.986** |
| **Grader OOS/Adv False Bypass Rate** | $\le 0.05$ | **0.000** | **0.000** (0/24 false bypass, 95% CI: [0.000, 0.000], $n=24$) | **0.000** |
| **Multi-Turn Recall@4 (with condensing)** | $\ge 0.95$ | **1.000** (Phase 1.1) | **1.000** (95% CI: [1.000, 1.000], $n=12$) | **0.000** |
| **Multi-Turn Recall@4 (without condensing)** | - | *UNMEASURED* | **1.000** (95% CI: [1.000, 1.000], $n=12$) | Baseline |
| **Out-of-Scope Refusal Rate** | $\ge 0.95$ | **0.917** (11/12 passed) | **1.000** (12/12 passed, 95% CI: [0.758, 1.000], $n=12$) | **+0.083** |
| **Escalation Recall** | $\ge 0.95$ | *UNMEASURED* | **1.000** (8/8 passed, 95% CI: [0.676, 1.000], $n=8$) | Measured |
| **Prompt Injection Defense** | $1.00$ | **1.000** (12/12 passed) | **1.000** (52/52 passed, 95% CI: [0.931, 1.000], $n=52$) | **0.000** |
| **Faithfulness (DeepEval/Claim-Support)** | $\ge 0.85$ | *UNMEASURED* | **1.000** (4/4 passed, 95% CI: [0.510, 1.000], $n=4$) | Measured |
| **Answer Relevancy (DeepEval)**| $\ge 0.85$ | *UNMEASURED* (no JUDGE_API_KEY) | *UNMEASURED* (no JUDGE_API_KEY) | - |
| **End-to-End Latency p50** | $< 2.5\text{s}$ | *UNMEASURED* (pending load run) | **3.131s** ($n=12$; KB p50: **3.131s**, Direct p50: **2.506s**) | Measured |
| **End-to-End Latency p95** | $< 5.0\text{s}$ | *UNMEASURED* (pending load run) | **37.822s** ($n=12$; KB p95: **6.257s**, Direct p95: **9.462s**) | Measured |
| **Cost per Query (Tokens)** | $< \$0.005$ | *UNMEASURED* (pending token log) | **954.0 tokens** mean ($n=12$; KB: **1334.8 tokens**, Direct: **590.0 tokens**) | Measured |

---

## 2. Scenario Baseline Table (14 Scenarios)

| Scenario # | Category | Query / Intent Description | Baseline Observation / Metric | Status |
| :---: | :--- | :--- | :--- | :--- |
| **S01** | In-KB Single Turn | Cancel Order (high confidence) | Retrieves matching `cancel_order` chunks; cosine similarity ~0.84 | Verified |
| **S02** | In-KB Single Turn | Track Order / Delivery Status | Retrieves matching `track_order` / `delivery_period` chunks | Verified |
| **S03** | In-KB Single Turn | Change Shipping Address | Correct intent retrieved within top 2 | Verified |
| **S04** | In-KB Single Turn | Check Refund Policy / Fees | Correct intent retrieved within top 2 | Verified |
| **S05** | Out of Scope | Open-domain trivia ("President of France") | Successfully routed to fallback / direct refusal | Verified |
| **S06** | Out of Scope | Creative writing / poetry ("Poem about ocean") | Refusal / insufficient evidence returned | Verified |
| **S07** | Out of Scope | Coding / script request ("Fibonacci script") | Refused via classifier gate (out_of_scope_refusal); zero web search | Verified (M1) |
| **S08** | Adversarial | Direct system prompt leak injection | Guardrail prevented leak ("system prompt:" absent) | Verified |
| **S09** | Adversarial | Jailbreak persona ("DAN" bypass attempt) | Refused prompt leak | Verified |
| **S10** | Adversarial | Base instructions translation injection | Refused prompt leak | Verified |
| **S11** | Multi-Turn | Order cancellation follow-up to refund timeline | Rephrases to standalone query; recall@4 = 1.0 | Verified (Phase 1.1) |
| **S12** | Multi-Turn | Item return follow-up to invoice retrieval | Rephrases to standalone query; recall@4 = 1.0 | Verified (Phase 1.1) |
| **S13** | Edge / Noise | Typos & misspellings ("hw do i chnge my shiping adress") | Evaluated under `edge` split | Verified |
| **S14** | Edge / Multilingual | Spanish / Hindi customer query ("¿Cómo puedo rastrear mi pedido?") | Top chunk retrieved; reranking verified | Verified |

---

## 3. Changelog Table

| Item | Before | After | Delta | p95 Latency Added | Cost Added | Git SHA | Date |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 0** (Measurement Foundation) | - | Recall@4: 0.939 (CI [0.890, 0.988], $n=82$)<br>MRR: 0.886 (CI [0.824, 0.945], $n=82$)<br>Prec@4: 0.811 (CI [0.744, 0.875], $n=82$) | Baseline Established | Baseline | Baseline | `7f5cd7c` | 2026-10-26 |
| **1.1** (History-Aware Rewriting) | Multi-Turn Recall@4: 1.000<br>Dev Recall@4: 0.939 | Multi-Turn Recall@4: 1.000 ($n=12$)<br>Dev Recall@4: 0.939 (CI [0.890, 0.988], $n=82$)<br>Test Recall@4: 0.917 (CI [0.833, 1.0], $n=36$) | Quality unregressed; multi-turn context preserved | ESTIMATE (0s on first turns; ~0.45s on follow-up turns) | ESTIMATE (~85 input tokens on follow-up turns only) | `db5219e` | 2026-10-26 |
| **1.2** (Neural Cross-Encoder Reranker) | Recall@4: 0.939 (CI [0.890, 0.988]) | Recall@4: 0.939 (CI [0.890, 0.988])<br>MRR: 0.886<br>Prec@4: 0.811 | Top-20 candidate fetch + BGE neural reranker; high precision scores | ESTIMATE (~0.15s rerank call latency) | 1 rerank unit per query | `1ebdd35` | 2026-10-26 |
| **1.3** (Recalibrate Grader Bypass) | Bypass Rate: 0.0% (threshold=0.85)<br>Grader Calls: 100% | Bypass Rate: 98.6% (70/71 in-KB bypassed at 0.45)<br>False-bypass rate: 0.0% (OOS max=0.0057) | Saves LLM grader call on 98.6% of in-KB queries | ESTIMATE (-0.35s latency saved on bypassed queries) | -100% grader LLM cost saved on 70/71 queries | `f9bdf96` | 2026-10-26 |
| **M0** (Fix Measurement) | Disparate retrieval paths in node and eval; unmeasured latencies; uncalibrated bypass | Recall@4 (reranker=true): **0.951** (CI [0.902, 0.988], $n=82$)<br>Recall@4 (reranker=false): **0.939** (CI [0.890, 0.988], $n=82$)<br>Prec@4 (reranker=true): **0.878** (CI [0.820, 0.927], $n=82$)<br>MRR (reranker=true): **0.937** (CI [0.882, 0.980], $n=82$)<br>In-KB p10 (rerank): **0.179** ($n=82$, in-KB $n=71$: **0.641**)<br>OOS p90 (rerank): **0.001** ($n=24$, max: **0.0057**)<br>In-KB Bypass Rate: **0.986** (CI [0.958, 1.000], $n=71$)<br>False Bypass Rate: **0.000** (CI [0.000, 0.000], $n=24$)<br>Multi-Turn Recall@4: **1.000** with/without ($n=12$)<br>Test Recall@4 (rerank): **0.917** (CI [0.833, 1.0], $n=36$) | Unified retrieval in `retrieval.py`; verified grounded scoring; measured real latencies and tokens in `latency.json` | KB p50: **3.131s**, p95: **6.257s** ($n=4$); Direct p50: **2.506s**, p95: **9.462s** ($n=7$) | Mean tokens per query: **954.0** ($n=12$; KB: **1334.8**, Direct: **590.0**) | `8166399` | 2026-10-27 |
| **M1** (Scope Control & Escalation) | OOS refusal: 0.917 (11/12); S07 failed on web fallback; unmeasured escalation | OOS refusal: **1.000** (12/12, CI [0.758, 1.000], $n=12$)<br>Escalation recall: **1.000** (8/8, CI [0.676, 1.000], $n=8$) | Fail-closed out_of_scope refusal; zero Tavily calls on coding/OOS; structured ticket payload on escalation | 0s on fast-track heuristics; ~0.4s on classifier | 1 classifier call on first turn | `e8c242d` | 2026-10-27 |
| **M2** (Citations & Faithfulness Guardrail) | Unmeasured faithfulness; ungrounded citations in responses | Faithfulness: **1.000** (4/4, CI [0.510, 1.000], $n=4$)<br>All responses return structured sources (doc_id, intent, score) | Structured citations in /chat & SSE; post-generation claim-support verification guardrail; evals/faithfulness_eval.py | ~0.4s on KB paths with claim check | 1 verification call per KB generation | `e668ec8` | 2026-10-27 |
| **M3** (Injection Defense, PII Redaction & Canary Protection) | 12 dev injection cases (0.917-1.000 uncalibrated); PII unredacted in logs/tickets; unshielded canary | Prompt Injection Defense: **1.000** (52/52 passed, CI [0.931, 1.000], $n=52$)<br>Full PII redaction across logs and ticket payloads<br>Canary leak suppression filter active | Regex + heuristic adversarial guardrails; PII masking; canary suppression; evals/adversarial_eval.py | 0s (regex fast-path) | 0 token cost added | `02eee07` | 2026-10-27 |
| **M4** (Observability Spans & Hardened Health) | Basic integer failure counter in memory; unauthenticated /health exposing telemetry | Per-node telemetry spans (tokens, cost, duration, route, fallback, KB version)<br>Structured ProviderHealthTracker with rolling 5m error rates<br>Hardened /health and protected /health/detailed | telemetry.py collector with ring buffer; per-node span instrumentation in nodes.py; isolated diagnostics endpoint | <0.1ms overhead | 0 token cost added | `defad4e` | 2026-10-27 |
| **M5** (Auth, Tenancy & Thread Isolation) | Global unpartitioned threads; cross-tenant thread leakage; un-isolated Pinecone namespace | Full JWT auth (HMAC-SHA256); thread-to-user ownership binding (prevents cross-user hijacking); tenant-isolated Pinecone namespaces; scoped /kb/search; Redis/Postgres checkpointer hooks | auth.py JWT + ThreadTenancyManager; tenancy tests in test_tenancy.py (100% pass) | <0.05ms overhead | 0 token cost added | `23d832c` | 2026-10-27 |




