# Security Policy & Architecture

## Reporting Vulnerabilities

If you discover a security vulnerability in this project, please report it responsibly. **Do NOT report security vulnerabilities through public GitHub issues.**

Please report security concerns via GitHub Private Vulnerability Reporting:
- **GitHub Security Advisories**: Navigate to the repository's **Security** tab > **Advisories** > **Report a vulnerability** ([https://github.com/tilakchow/agentic-customer-support-rag/security/advisories](https://github.com/tilakchow/agentic-customer-support-rag/security/advisories))
- Subject: `[SECURITY] Agentic RAG Vulnerability Report`

Please include:
1. Description of the vulnerability and attack vector.
2. Steps to reproduce or proof-of-concept payload.
3. Impact assessment.

We strive to acknowledge receipt within 48 hours and provide a remediation timeline.

---

## Security Architecture & Defenses

### 1. Scope Control & Fail-Closed Gatekeeping
- The `classify_scope` node acts as an initial gatekeeper before retrieval or tool execution.
- Disallowed queries (programming scripts, code generation, creative fiction, general web trivia, and hostile injections) are rejected immediately via `refuse_out_of_scope`.
- External search engines (Tavily) are never called for out-of-scope or coding queries, preventing resource exhaustion and third-party credential leakage.

### 2. Multi-Tenant Isolation & Thread Ownership Protection
- All chat requests extract authenticated `user_id` and `tenant_id` claims from signed JWTs (`Authorization: Bearer <token>`).
- Conversation threads are registered and bound to the creating tenant and user via `ThreadTenancyManager`. Cross-user or cross-tenant attempts to post to another caller's `thread_id` are rejected with `403 Forbidden`.
- Pinecone vector queries are partitioned per tenant: `{tenant_id}-{namespace}` ensures complete isolation of tenant knowledge bases.

### 3. PII Redaction & Data Sanitization
- Incoming user inquiries pass through `redact_pii()` before being stored in state checkpoints, error logs, or customer support escalation tickets.
- Payment card numbers (Luhn candidate strings), Social Security Numbers (SSNs), phone numbers, email addresses, and API credentials (`gsk_...`, `pcsk_...`, `tvly-...`, `Bearer ...`) are automatically masked with structured `[REDACTED_*]` tokens.
- Logger statements at `INFO` level record query lengths rather than raw customer queries. Detailed queries are logged only at `DEBUG` level with active PII masking.

### 4. Prompt Injection & Canary Token Leakage Protection
- Hostile prompt injection signatures (DAN personas, system override commands, base instruction translation, delimiter breakouts, `<|im_start|>`, `[SYSTEM]`) are intercepted deterministically.
- An internal security canary token (`CANARY_GUARD_7f9c2d1b`) is embedded within internal system instructions.
- The `filter_canary_and_leaks()` guardrail monitors token streams and completed model outputs, suppressing any canary token or system prompt text before transmission to clients.

### 5. Post-Generation Faithfulness & Citation Verification
- Responses generated from internal documents are audited by `check_claim_support()`, verifying that every factual claim is grounded in retrieved knowledge base chunks.
- Every response returned over `/chat` or SSE includes structured source citations (`doc_id`, `intent`, `score`) to ensure auditability.

### 6. Trusted Reverse Proxy Configuration
- By default, the rate limiter does **not** trust `X-Forwarded-For` headers to prevent client IP spoofing attacks.
- When running behind an enterprise reverse proxy (e.g., Nginx, Cloudflare, AWS ALB, Render):
  - Set `TRUSTED_PROXIES` in your environment (comma-separated list of proxy IPs/CIDRs):
    ```env
    TRUSTED_PROXIES=10.0.0.1,172.16.0.1
    ```
  - The backend evaluates the proxy chain right-to-left and extracts the nearest untrusted client IP.

### 7. Hardened Operational Probes & Metrics
- Public `/health` endpoints act as simple liveness/readiness probes, exposing only high-level status without internal error strings or stack traces.
- Granular operational telemetry, per-provider error rates, and node execution spans are isolated behind authentication at `/health/detailed` requiring `API_KEY` verification.

### 8. Production Checkpointer Isolation
- Ephemeral in-memory checkpointers (`MemorySaver`) store state indefinitely in process memory, leading to unbounded memory growth.
- In production (`ENVIRONMENT=production`), the application **refuses to boot** unless:
  - `CHECKPOINT_DB` points to a persistent database (e.g. `checkpoints.sqlite`), or
  - `POSTGRES_URL` is configured with `AsyncPostgresSaver`, or
  - `ALLOW_IN_MEMORY_CHECKPOINTS=true` is explicitly configured for ephemeral testing.
- State per conversation thread is bounded to `MAX_THREAD_MESSAGES` (default 40).
