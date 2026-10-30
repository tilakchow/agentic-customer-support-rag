<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/hero-dark.svg">
    <img alt="Agentic Customer Support RAG Banner" src="assets/hero.svg" width="100%">
  </picture>
</p>

<p align="center">
  <b>Production-grade, self-correcting customer support agent built with LangGraph, FastAPI, Pinecone, and React 19.</b>
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.12"></a>
  <a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://langchain.com/langgraph"><img src="https://img.shields.io/badge/LangGraph-1.2+-4F46E5?style=flat-square&logo=langchain&logoColor=white" alt="LangGraph"></a>
  <a href="https://www.pinecone.io"><img src="https://img.shields.io/badge/Pinecone-Serverless-000000?style=flat-square&logo=pinecone&logoColor=white" alt="Pinecone"></a>
  <a href="https://react.dev"><img src="https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black" alt="React 19"></a>
  <a href="tests/"><img src="https://img.shields.io/badge/Tests-241%20Passing-10B981?style=flat-square&logo=pytest&logoColor=white" alt="Tests"></a>
  <a href="https://github.com/tilakchow/agentic-customer-support-rag/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/tilakchow/agentic-customer-support-rag/ci.yml?branch=main&style=flat-square&logo=githubactions&logoColor=white&label=CI" alt="CI Status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-334155?style=flat-square" alt="License MIT"></a>
</p>

<p align="center">
  <a href="https://agentic-customer-support-rag.vercel.app/">🚀 <b>Live Demo</b></a> ·
  <a href="#-results">📊 <b>Scoreboard</b></a> ·
  <a href="#-how-it-works">🏗 <b>Architecture</b></a> ·
  <a href="#-security--guardrails">🔐 <b>Security</b></a> ·
  <a href="#-quick-start">⚡ <b>Quick Start</b></a>
</p>

> **Note:** Backend runs on a free Render instance and may take ~30-50s to wake up on first request.

---

## ✨ What it does

An autonomous customer support agent that replaces fragile single-prompt RAG with a self-correcting LangGraph state machine. It verifies query safety before search, reranks vector candidates with a neural cross-encoder, grades evidence relevance, validates generated claims against source documents, and escalates to human staff when confidence is low.

| 🛡 Safety | 🎯 Retrieval | ✅ Grounding | 🧰 Tools |
| :--- | :--- | :--- | :--- |
| Pre-retrieval scope gating, prompt injection defense, and PII masking | Neural cross-encoder reranking with multi-turn history condensation | Post-generation claim checks and structured source citations | Tenant-scoped order lookups and automated support ticket creation |

---

## 🎬 Demo

| Chat answer with sources | Automated ticket escalation | Out-of-scope refusal |
| :---: | :---: | :---: |
| <img src="assets/demo-answer.png" width="100%" alt="Answer with sources"> | <img src="assets/demo-escalation.png" width="100%" alt="Ticket escalation"> | <img src="assets/demo-refusal.png" width="100%" alt="Scope refusal"> |
| *Grounded KB answer with citations* | *Legal/dissatisfaction escalation* | *Off-topic & prompt injection defense* |

---

## 🧠 How it works

The LangGraph decision loop dynamically routes every message through security checks, retrieval, grading, and post-generation verification:

```mermaid
flowchart TD
    U([User message]) --> S{Scope check}
    S -->|injection or off-topic| R[Refuse]
    S -->|human request or legal threat| E[Escalate: ticket]
    S -->|classifier failed| NS[Not sure]
    S -->|in scope| RQ[Route question]
    RQ --> PT[Plan tools: orders, tickets]
    PT -->|greeting or small talk| D[Direct answer]
    PT -->|support question| C[Condense query]
    C --> RK[Retrieve + rerank]
    RK --> G{Grade evidence}
    G -->|good| GK[Generate from KB + claim check]
    G -->|weak: web off, contact, policy or blocked topic| NS
    G -->|weak: web on| W[Search web - Tavily]
    W --> GW{Grade web results}
    GW -->|good| GWB[Generate from web]
    GW -->|weak, retries left| RW[Rewrite query]
    RW --> RK
    GW -->|retries over or search failed| NS
    GK --> A([Answer with sources])
    GWB --> A
    D --> A
    R --> A
    E --> A
    NS --> A

    classDef gate fill:#EEF2FF,stroke:#4F46E5,color:#1E1B4B,stroke-width:1.5px
    classDef step fill:#CCFBF1,stroke:#0D9488,color:#134E4A
    classDef optional fill:#FEF3C7,stroke:#D97706,color:#78350F,stroke-dasharray:4 3
    classDef stop fill:#F1F5F9,stroke:#64748B,color:#1E293B
    class S,G,GW gate
    class RQ,PT,C,RK,GK,RW step
    class W,GWB optional
    class R,E,NS,D stop
```

<p align="center">
  <sub><b>Legend:</b> <span style="color:#4F46E5">■</span> Indigo = decision gate &nbsp;|&nbsp; <span style="color:#0D9488">■</span> Teal = work step &nbsp;|&nbsp; <span style="color:#D97706">■</span> Amber dashed = optional (web fallback, off by default) &nbsp;|&nbsp; <span style="color:#64748B">■</span> Slate = terminal stop</sub>
</p>

### Why each path is taken

- **Refuse (`refuse_out_of_scope`):** Triggered when inbound input attempts prompt injection, system prompt exfiltration, coding requests, or open-domain trivia. Fails closed before retrieval.
- **Escalate (`escalate_to_human`):** Triggered by litigation threats, persistent complaints, or explicit agent requests. Generates a structured ticket (`TICK-XXXXXXXX`) and redacts PII.
- **Direct answer (`direct_answer`):** Short-circuits greetings and conversational pleasantries without vector search overhead.
- **Condense (`condense_query`):** Reformulates multi-turn context (e.g., "what about shipping?") into a standalone search query.
- **Grader bypass (`grade_kb_evidence`):** If cross-encoder score $\ge 0.45$, the LLM evidence grader is skipped (applies to 98.6% of in-KB queries). Low-scoring evidence is evaluated by LLM.
- **Web search fallback (`search_web`):** Disabled by default (`WEB_FALLBACK_ENABLED=false`). When enabled, executes only on ambiguous queries that pass policy and legal blacklists.
- **Claim check & cheap pre-check (`generate_from_kb`):** Extracts numbers, URLs, emails, currencies, and phone numbers before running LLM verification. If all extracted entities match the retrieved context, the expensive LLM check is bypassed, saving ~800 tokens on verified grounded answers.
- **Classifier history truncation:** Limits context passed to `classify_scope`, `route_question`, and tool planning to the last 4 turns with assistant messages truncated to 200 characters, saving ~50-80% input tokens in multi-turn dialogues.
- **Context preamble stripping:** Strips redundant `User Request:` prompt templates from retrieved KB chunks, passing only the recommended response text to the generator (~72 tokens/query saved).
- **Hybrid BM25 + Dense retrieval (`retrieve`):** Evaluated in-memory BM25 with Reciprocal Rank Fusion ($k=60$) behind `HYBRID_RETRIEVAL_ENABLED` (default `false` as dense baseline demonstrated higher precision on customer support benchmarks).

<details>
<summary><b>View complete 15-node execution table & fail modes</b></summary>

- Full node role breakdown: [`docs/NODES.md`](docs/NODES.md)
- Node-by-node failure policies & fail-open/closed matrix: [`docs/FAILURE_BEHAVIOR.md`](docs/FAILURE_BEHAVIOR.md)
</details>

---

## 📊 Results

Empirical results from automated evaluation suites. Evaluated against the Bitext customer support dataset with synthetic augmentations. Sample sizes are noted per split ($n$). Full historical logs, confidence intervals, and baselines are maintained in [`evals/SCOREBOARD.md`](evals/SCOREBOARD.md).

| Evaluation Metric | Target | Result (Empirical) | Status |
| :--- | :---: | :---: | :---: |
| **Recall@4 (Dev split, reranker=true)** | $\ge 0.95$ | **0.951** (95% CI: [0.902, 0.988], $n=82$) | ✅ Passing |
| **Recall@4 (Dev split, reranker=false)** | - | **0.939** (95% CI: [0.890, 0.988], $n=82$) | ✅ Passing |
| **Recall@4 (Test split, reranker=true)** | $\ge 0.90$ | **0.917** (95% CI: [0.833, 1.000], $n=36$) | ✅ Passing |
| **Precision@4 (Dev split, reranker=true)** | $\ge 0.85$ | **0.878** (95% CI: [0.820, 0.927], $n=82$) | ✅ Passing |
| **MRR (Dev split, reranker=true)** | $\ge 0.90$ | **0.937** (95% CI: [0.882, 0.980], $n=82$) | ✅ Passing |
| **Grader In-KB Bypass Rate (score $\ge 0.45$)** | $\ge 0.90$ | **0.986** (70/71 bypassed, 95% CI: [0.958, 1.000], $n=71$) | ✅ Passing |
| **Grader False Bypass Rate (OOS / Adversarial)** | $\le 0.05$ | **0.000** (0/24 bypassed, 95% CI: [0.000, 0.000], $n=24$) | ✅ Passing |
| **Multi-Turn Recall@4 (History Condensing)** | $\ge 0.95$ | **1.000** (95% CI: [1.000, 1.000], $n=12$) | ✅ Passing |
| **Out-of-Scope Refusal Rate** | $\ge 0.95$ | **1.000** (12/12 refused, 95% CI: [0.758, 1.000], $n=12$) | ✅ Passing |
| **Escalation Recall** | $\ge 0.95$ | **1.000** (8/8 escalated, 95% CI: [0.676, 1.000], $n=8$) | ✅ Passing |
| **Prompt Injection Defense** | $1.00$ | **1.000** (52/52 protected, 95% CI: [0.931, 1.000], $n=52$) | ✅ Passing |
| **Cheap Faithfulness Pre-Check Accuracy** | $\ge 0.90$ | **1.000** (24/24 cases, 18/18 hallucinations caught, $n=24$) | ✅ Passing |
| **Answer Groundedness (LLM Judge)** | $\ge 4.5$ | **5.000 / 5.0** (95% CI: [5.000, 5.000], $n=20$) | ✅ Passing |
| **Answer Tone (LLM Judge)** | $\ge 4.5$ | **5.000 / 5.0** (95% CI: [5.000, 5.000], $n=20$) | ✅ Passing |
| **Answer Correctness (LLM Judge)** | $\ge 3.5$ | **3.700 / 5.0** (95% CI: [3.150, 4.300], $n=20$) | ✅ Passing |
| **Tokens per Query (T0 Baseline vs T1-T5)** | $< \$0.005$ | **1,742.1 mean** (-26.6% from 2,372.2 baseline; $n=40$ stratified) | ✅ Passing |
| **System Prompt Compression** | $\le 260\text{ tok}$ | **258 tokens** (down from 462, **-44.2%** static overhead) | ✅ Passing |
| **Operational Tool Benchmark** | $\ge 0.95$ | **1.000** (20/20 orders & tickets, $n=20$) | ✅ Passing |
| **End-to-End Latency p50** | $< 2.5\text{s}$ | **3.131s** ($n=12$; KB: 3.131s, Direct: 2.506s) | ❌ Missed |
| **End-to-End Latency p95** | $< 5.0\text{s}$ | **37.822s** ($n=12$; KB: 6.257s, Direct: 9.462s) | ❌ Missed |


### 📉 Token & Cost Optimization Breakdown (T0 Baseline vs Optimized)

Measured using live telemetry on a 40-query stratified dev split (`evals/tokens.json`). Optimized runs incorporate role-specific reasoning caps (T1), classifier history truncation (T2), cheap entity pre-checks (T3), and prompt compression (T4).

| Execution Path | Baseline Tokens (Mean) | Baseline Calls | Optimized Tokens (Mean) | Optimized Calls | Token Delta (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Knowledge Base (KB)** | 5,230.5 | 5.00 | **4,802.6** | 5.71 | **-8.2%** |
| **Operational Tools** | 2,716.9 | 4.12 | **2,533.5** | 4.12 | **-6.7%** |
| **Direct Answer (Chit-chat)** | 1,464.8 | 3.00 | **1,427.2** | 3.00 | **-2.6%** |
| **Insufficient Evidence** | 2,493.8 | 4.20 | **895.3** | 1.67 | **-64.1%** |
| **Escalate to Human** | 1,310.4 | 1.25 | **128.7** | 0.29 | **-90.2%** |
| **Out-of-Scope Refusal** | 172.5 | 0.33 | **160.3** | 0.33 | **-7.1%** |
| **Query-Wide Average** | **2,372.2** | **3.02** | **1,742.1** | **2.62** | **-26.6% (-630 tok/q)** |


---

## 🧱 Tech stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Orchestration** | ![LangGraph](https://img.shields.io/badge/LangGraph-1.2+-4F46E5?style=flat-square&logo=langchain&logoColor=white) | Cyclic state machine, thread checkpoints, and conditional routing |
| **API Framework** | ![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white) | Async REST API, connection semaphores, and Server-Sent Events (SSE) |
| **Vector DB** | ![Pinecone](https://img.shields.io/badge/Pinecone-Serverless-000000?style=flat-square&logo=pinecone&logoColor=white) | Managed vector indexing with tenant-partitioned namespaces |
| **Embeddings** | ![FastEmbed](https://img.shields.io/badge/FastEmbed-all--MiniLM--L6--v2-2563EB?style=flat-square) | Local ONNX runtime producing 384-dimensional dense vectors |
| **Reranker** | ![BGE](https://img.shields.io/badge/BGE-reranker--v2--m3-D97706?style=flat-square) | Neural cross-encoder rescoring top-20 candidates to top-4 |
| **LLM Inference** | ![Groq](https://img.shields.io/badge/Groq-openai/gpt--oss--20b-F59E0B?style=flat-square) | Fast streaming inference with exponential retry backoff |
| **Web Search** | ![Tavily](https://img.shields.io/badge/Tavily-Search_API-475569?style=flat-square) | Optional fallback search (`WEB_FALLBACK_ENABLED=false` by default) |
| **Frontend** | ![React](https://img.shields.io/badge/React-19.2-61DAFB?style=flat-square&logo=react&logoColor=black) | Vite SPA, real-time SSE token streaming, light/dark theme |
| **Auth & Security** | ![JWT](https://img.shields.io/badge/JWT-HMAC--SHA256-E11D48?style=flat-square) | Scoped authentication, thread ownership, and tenant namespace isolation |
| **Observability** | ![Telemetry](https://img.shields.io/badge/Telemetry-Per--Node_Spans-6366F1?style=flat-square) | In-memory span collector, rolling error tracker, and isolated `/health` |

---

### Environment configuration reference

| Variable | Default | Description |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | `""` | Groq API key for LLM generation and classification |
| `PINECONE_API_KEY` | `""` | Pinecone API key for vector storage and retrieval |
| `TAVILY_API_KEY` | `""` | Optional Tavily API key for web search fallback |
| `INDEX_NAME` | `industry-agentic-rag-kb` | Pinecone vector index name |
| `NAMESPACE` | `bitext-support-v2` | Default vector namespace |
| `REASONING_EFFORT_SCOPE` | `low` | Groq reasoning effort cap for scope classifier (`low`, `medium`, `high`) |
| `MAX_TOKENS_SCOPE` | `None` | Output token cap for scope classifier |
| `REASONING_EFFORT_ROUTER` | `low` | Groq reasoning effort cap for question router |
| `MAX_TOKENS_ROUTER` | `None` | Output token cap for question router |
| `REASONING_EFFORT_GRADER` | `low` | Groq reasoning effort cap for evidence grader |
| `MAX_TOKENS_GRADER` | `None` | Output token cap for evidence grader |
| `REASONING_EFFORT_FAITHFULNESS` | `low` | Groq reasoning effort cap for claim-support check |
| `MAX_TOKENS_FAITHFULNESS` | `None` | Output token cap for claim-support check |
| `CLASSIFIER_HISTORY_MESSAGES`| `4` | Max conversation turns passed to classifier nodes |
| `CLASSIFIER_ASSISTANT_CHAR_LIMIT`| `200` | Max character length for assistant turns in classifier prompts |
| `HYBRID_RETRIEVAL_ENABLED` | `false` | Enable BM25 + Dense Reciprocal Rank Fusion ($k=60$) |
| `RRF_K` | `60` | Constant $k$ for Reciprocal Rank Fusion |
| `WEB_FALLBACK_ENABLED` | `false` | Enable external web search when KB evidence is weak |
| `SKIP_GRADER_SCORE` | `0.45` | Similarity threshold to bypass LLM evidence grader |


---

## 🚀 Quick start

<details open>
<summary><b>1. Backend (FastAPI + LangGraph)</b></summary>

```bash
# Clone and enter repo
git clone https://github.com/tilakchow/agentic-customer-support-rag.git
cd agentic-customer-support-rag

# Install dependencies with uv
uv sync

# Configure environment
cp .env.example .env
# Fill in GROQ_API_KEY, PINECONE_API_KEY, and JWT_SECRET in .env

# Run test suite (241 tests)
uv run pytest tests -q

# Start FastAPI server
uv run uvicorn agenticrag.api:app --host 0.0.0.0 --port 8000 --reload
```

**Query with `curl` (REST):**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I return an item?", "thread_id": "thread-101"}'
```

**Query with `curl` (SSE streaming):**
```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the refund timeline?", "thread_id": "thread-101"}'
```
</details>

<details>
<summary><b>2. Frontend (React 19 + Vite)</b></summary>

```bash
cd frontend
npm install
npm run dev
```

*Note for production:* Set `BACKEND_URL` in Vercel environment variables, then redeploy. The frontend proxies API requests through serverless routes to prevent browser mixed-content and cross-origin blocking.
</details>

<details>
<summary><b>3. Docker</b></summary>

```bash
# Run backend via Docker
docker build -t agentic-rag-backend .
docker run -p 8000:8000 --env-file .env agentic-rag-backend
```
</details>

---

## 🗂 Project structure

```text
agentic-customer-support-rag/
├── src/agenticrag/          # Core backend package
│   ├── api.py               # FastAPI endpoints (/chat, /chat/stream, /health, /tools)
│   ├── auth.py              # JWT authentication & thread-to-user tenancy manager
│   ├── cache.py             # Exact-match response cache (first-turn only) & LRU store
│   ├── clients.py           # Lazy client initializers (Groq, Pinecone, FastEmbed)
│   ├── config.py            # Pydantic v2 application settings & fail-mode policies
│   ├── graph.py             # LangGraph state machine & conditional edge definitions
│   ├── ingest.py            # Knowledge base ingestion with chunk hash-diffing
│   ├── nodes.py             # 15 graph node handlers (scope, condense, grade, generate)
│   ├── redaction.py         # Regex PII detection and masking engine
│   ├── retrieval.py         # Vector similarity search & BGE neural cross-encoder
│   ├── routing.py           # Scope classification, prompt injection & escalation logic
│   ├── security.py          # Canary token verification & sanitization filters
│   ├── state.py             # TypedDict agent state & message dataclasses
│   ├── telemetry.py         # Distributed per-node span recorder & rolling health metrics
│   └── tools.py             # Deterministic order status & support ticket tools
├── evals/                   # Evaluation harnesses & scoreboard
│   ├── SCOREBOARD.md        # Definitive metric tracker with confidence intervals
│   ├── token_report.py      # Per-node & per-path token distributions & LLM call telemetry
│   ├── hallucination_eval.py # Grounding entity pre-check & hallucination benchmark
│   ├── hybrid_eval.py       # BM25 + Dense RRF fusion evaluation on hard queries
│   ├── retrieval_eval.py    # Recall@4, Precision@4, MRR, and grader bypass sweep
│   ├── adversarial_eval.py  # 52 prompt injection and jailbreak attack tests
│   ├── answer_quality_eval.py # LLM-as-a-judge groundedness, tone, and correctness
│   └── tool_use_eval.py     # 20 operational order and ticket benchmarks
├── frontend/                # React 19 single-page application
│   ├── src/                 # Chat interface, source attribution drawer, theme toggle
│   └── api/                 # Vercel serverless edge proxy endpoints
├── docs/                    # Deep-dive documentation
│   ├── NODES.md             # Complete 15-node execution specification
│   └── FAILURE_BEHAVIOR.md  # Detailed fail-open/closed policies and timeout behavior
└── tests/                   # 241 automated pytest unit and integration tests
```

---

## 🔐 Security & guardrails

- **JWT Authentication & Tenancy:** Every request extracts `tenant_id` and `user_id` from signed JWT tokens to prevent session hijacking and cross-tenant leakage.
- **Tenant-Partitioned Vector Namespaces:** Pinecone queries and ingestion are strictly partitioned by tenant ID namespaces.
- **Thread Ownership Enforcement:** `ThreadTenancyManager` verifies that existing conversation threads cannot be accessed or modified by unauthorized users.
- **Automated PII Redaction:** Phone numbers, credit cards, emails, and SSNs are masked before persistence, logging, or ticket creation.
- **Pre-Retrieval Scope & Injection Filtering:** Fast regex heuristics and LLM classification block jailbreaks and prompt injections before vector retrieval.
- **Canary String Suppression:** A dynamic canary token (`CANARY_TOKEN`) in internal system prompts is actively stripped from all outbound streaming chunks.
- **Configurable Node Fail Modes:** Upstream timeouts or errors trigger explicit fail-closed (`classify_scope`, `grade_evidence`) or fail-open (`route_question`) behaviors.
- **Exact-Match Response Cache Partitioning:** The first-turn response cache is partitioned by `(kb_version, tenant_id, query_hash)` to prevent cross-tenant cache bleeding.

*For complete threat models and vulnerability disclosure instructions, see [`SECURITY.md`](SECURITY.md).*

---

## ⚠️ Known limitations

- **Synthetic Evaluation Dataset:** Benchmarks are measured on Bitext customer support data augmented with synthetic queries. Real enterprise distributions may vary.
- **Evaluated Sample Counts:** While core retrieval ($n=82/36$), adversarial ($n=52$), hallucination cases ($n=24$), answer quality ($n=20$), and tool suites ($n=20$) are systematically evaluated, smaller edge splits (e.g. escalation $n=8$, multi-turn $n=12$) have wider confidence intervals.
- **Heuristic Injection Defense:** Regex and LLM classifier filters provide defense-in-depth but cannot mathematically guarantee 100% protection against novel adversarial jailbreaks.
- **Web Fallback Lacks Claim Verification:** Responses generated via Tavily web search bypass the post-generation claim-support verification applied to KB answers.
- **In-Memory Tool Stores:** The mock order database and support ticket registry run in memory; production requires persistent PostgreSQL/Redis backing.
- **Cold Starts & Upstream Rate Limits:** Free-tier Render instances experience initial ~30-50s spin-up delays, and Groq on-demand enforces an 8,000 TPM limit.
- **Missed Latency Targets:** End-to-end p50 latency is 3.13s (target <2.5s) and p95 is 37.82s (target <5.0s) due to sequential multi-node validation and cold start variance.

---

## 🗺 Roadmap

- [x] LangGraph self-correcting state machine with multi-turn query condensation
- [x] BGE neural cross-encoder reranking with 98.6% grader bypass calibration
- [x] Pre-retrieval scope filter and prompt injection defense (52/52 passing)
- [x] Post-generation claim-support verification for knowledge base responses
- [x] Multi-tenant Pinecone namespace isolation and JWT thread tenancy
- [x] Exact-match response cache (first-turn only) with automatic KB version invalidation
- [x] Hybrid retrieval experiment (In-memory BM25 + Dense RRF $k=60$ evaluated on 42 hard cases; tried, no measured gain over dense-only)
- [ ] Automated post-generation claim checks for web fallback answers
- [ ] Persistent PostgreSQL & Redis backends for orders, tickets, and thread checkpointers
- [ ] Held-out adversarial benchmark suite ($n \ge 500$)
- [ ] Distributed OpenTelemetry tracing dashboard with live LangSmith / Jaeger export
- [ ] Larger realistic customer evaluation dataset ($n \ge 500$)

---

## 👤 Author

Built by **Tilak** ([@tilakchow](https://github.com/tilakchow))  
<!-- TODO: Add LinkedIn URL -->
<!-- TODO: Add X/Twitter URL -->

---

## 📄 License

This project is licensed under the MIT License - see the [`LICENSE`](LICENSE) file for details.
