# Security Policy

## Reporting Vulnerabilities

If you discover a security vulnerability in this project, please report it responsibly. **Do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report security concerns via GitHub Private Vulnerability Reporting:
- **GitHub Security Advisories**: Navigate to the repository's **Security** tab > **Advisories** > **Report a vulnerability** ([https://github.com/tilakchow/agentic-customer-support-rag/security/advisories](https://github.com/tilakchow/agentic-customer-support-rag/security/advisories))
- Subject: `[SECURITY] Agentic RAG Vulnerability Report`

Please include:
1. Description of the vulnerability and attack vector.
2. Steps to reproduce or proof-of-concept payload.
3. Impact assessment.

We strive to acknowledge receipt within 48 hours and provide a remediation timeline.

---

## Security Architecture & Best Practices

### 1. Trusted Reverse Proxy Configuration
By default, the rate limiter does **not** trust `X-Forwarded-For` headers to prevent client IP spoofing attacks.
When running behind an enterprise reverse proxy (e.g., Nginx, Cloudflare, AWS ALB, Render):
- Set `TRUSTED_PROXIES` in your environment (comma-separated list of proxy IPs/CIDRs):
  ```env
  TRUSTED_PROXIES=10.0.0.1,172.16.0.1
  ```
- The backend evaluates the proxy chain right-to-left and extracts the nearest untrusted client IP.

### 2. Checkpointer Persistence in Production
- Ephemeral in-memory checkpointers (`MemorySaver`) store state indefinitely in process memory, leading to unbounded memory growth.
- In production (`ENVIRONMENT=production`), the application **refuses to boot** unless:
  - `CHECKPOINT_DB` points to a persistent database (e.g. `checkpoints.sqlite`), or
  - `ALLOW_IN_MEMORY_CHECKPOINTS=true` is explicitly configured for ephemeral testing.
- State per conversation thread is bounded to `MAX_THREAD_MESSAGES` (default 40).

### 3. API Key & Client Secret Isolation
- Backend secrets (`GROQ_API_KEY`, `TAVILY_API_KEY`, `PINECONE_API_KEY`, `API_KEY`) must **never** be shipped in public browser bundles.
- If backend access requires authentication, configure `API_KEY` on the backend and route browser clients through the provided serverless edge proxy (`/api/chat` and `/api/chat/stream`), which injects `X-API-Key` securely on the server side.

### 4. Input Sanitization & Prompt Injection Mitigation
- Untrusted context retrieved from external search engines or vector stores is passed through `sanitize_untrusted_context` to strip prompt injection delimiter markers and system override sequences before LLM ingestion.
