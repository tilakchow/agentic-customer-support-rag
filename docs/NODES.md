# Graph Execution Nodes Reference

The LangGraph state machine orchestrates 15 discrete nodes to process support inquiries, manage conversational state, enforce security boundaries, and synthesize grounded responses.

| Node Name | Node Identifier | Category | Responsibilities & Behavior |
| :--- | :--- | :---: | :--- |
| **Classify Scope** | `classify_scope` | Security Gate | Pre-execution safety filter. Scans queries using heuristics and LLM classification to detect prompt injections, out-of-scope tasks (e.g. general trivia, code generation), and human escalation triggers before any retrieval occurs. |
| **Refuse Out-of-Scope** | `refuse_out_of_scope` | Terminal Stop | Gracefully refuses out-of-scope queries (e.g. coding requests, general trivia) without consuming external retrieval or web search tokens. |
| **Escalate to Human** | `escalate_to_human` | Terminal Stop | Automatically detects legal threats, severe customer dissatisfaction, or explicit human agent requests. Redacts PII and generates a structured support ticket payload (`TICK-XXXXXXXX`). |
| **Route Question** | `route_question` | Intent Gate | Routes inbound inquiries into conversational categories: `kb` (queries requiring knowledge base retrieval) or `direct` (social pleasantries, greetings). |
| **Plan Tools** | `plan_tools` | Operational | Evaluates whether operational customer support tools (e.g., `get_order_status`, `create_ticket`) are needed, enforces order tenancy boundaries, and executes tool calls before KB retrieval. |
| **Condense Query** | `condense_query` | Query Processing | Transforms conversational follow-up turns into standalone, context-complete search queries using conversation history, preserving multi-turn retrieval recall. |
| **Retrieve KB** | `retrieve_kb` | Retrieval | Embeds queries via FastEmbed (`all-MiniLM-L6-v2`), fetches top-20 candidates from Pinecone within tenant namespace, and applies BGE neural cross-encoder reranking (`bge-reranker-v2-m3`) to select top-4 chunks. |
| **Grade KB Evidence** | `grade_kb_evidence` | Evaluation Gate | Validates whether retrieved chunks contain sufficient evidence. High-confidence vector matches ($\ge 0.45$) automatically bypass LLM grading, saving latency and cost on 98.6% of in-KB queries. |
| **Search Web** | `search_web` | Fallback Retrieval | Fallback search via Tavily Search API. Executes only when `WEB_FALLBACK_ENABLED=true` and internal KB evidence is insufficient. Hard-blocked on policy, legal, or contact queries. |
| **Grade Web Evidence** | `grade_web_evidence` | Evaluation Gate | Evaluates whether web search snippets are relevant and trustworthy before passing them to generation. |
| **Rewrite Query** | `rewrite_query` | Query Processing | Reformulates query phrasing when retrieval evidence is ambiguous, allowing a single self-correcting retry loop. |
| **Generate from KB** | `generate_from_kb` | Generation | Synthesizes customer response strictly grounded in retrieved KB chunks, attributing structured citations (`doc_id`, `intent`, `score`), followed by a post-generation claim check. |
| **Generate from Web** | `generate_from_web` | Generation | Synthesizes customer response grounded in verified Tavily search snippets when web fallback is active. |
| **Direct Answer** | `direct_answer` | Direct Response | Emits immediate conversational answers for greetings, pleasantries, or system introductions without vector retrieval overhead. |
| **Answer Insufficient** | `answer_insufficient` | Terminal Stop | Transparently informs the customer that verified documentation is unavailable, offering helpful contact options rather than hallucinating facts. |
