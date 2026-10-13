export interface ChatRequest {
  message: string
  thread_id?: string
}

export interface ChatResponse {
  response: string
  thread_id: string
  source_used?: string
}

export interface HealthResponse {
  status: string
  service: string
}

export interface StreamEvent {
  type: "status" | "token" | "done" | "error"
  node?: string
  message?: string
  token?: string
  thread_id?: string
  source_used?: string
  full_content?: string
  error?: string
}

export interface StreamCallbacks {
  onStatus?: (message: string, node?: string) => void
  onToken?: (token: string) => void
  onDone?: (data: { thread_id: string; source_used?: string; full_content?: string }) => void
  onError?: (error: string) => void
}

// Client API configuration: Defaults to '/api' so all requests route through the
// reverse-proxy / edge functions (injecting auth and preventing direct exposure).
// If a custom backend or direct endpoint is explicitly desired, set VITE_API_URL.
const API_BASE = (import.meta.env.VITE_API_URL || "/api").replace(/\/$/, "");
const API_KEY = import.meta.env.VITE_API_KEY || "";

function getHeaders(extraHeaders: Record<string, string> = {}): Record<string, string> {
  const headers: Record<string, string> = { ...extraHeaders };
  if (API_KEY) {
    headers["X-API-Key"] = API_KEY;
  }
  return headers;
}

export async function streamMessage(req: ChatRequest, callbacks: StreamCallbacks): Promise<void> {
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: getHeaders({
      "Content-Type": "application/json",
    }),
    body: JSON.stringify(req),
  })

  if (!response.ok) {
    const errorData = await response.json().catch(() => null)
    throw new Error(errorData?.detail || `Streaming error: ${response.status} ${response.statusText}`)
  }

  if (!response.body) {
    throw new Error("ReadableStream not supported on response body")
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder("utf-8")
  let buffer = ""

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split("\n\n")
      // Retain last incomplete part in buffer
      buffer = lines.pop() || ""

      for (const line of lines) {
        const trimmed = line.trim()
        if (trimmed.startsWith("data: ")) {
          try {
            const event: StreamEvent = JSON.parse(trimmed.slice(6))
            if (event.type === "status" && callbacks.onStatus) {
              callbacks.onStatus(event.message || "", event.node)
            } else if (event.type === "token" && event.token && callbacks.onToken) {
              callbacks.onToken(event.token)
            } else if (event.type === "done" && callbacks.onDone) {
              callbacks.onDone({
                thread_id: event.thread_id || "",
                source_used: event.source_used,
                full_content: event.full_content,
              })
            } else if (event.type === "error" && callbacks.onError) {
              callbacks.onError(event.error || "Unknown error during streaming")
            }
          } catch (e) {
            console.warn("Failed to parse SSE chunk:", trimmed, e)
          }
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}

export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`, {
    headers: getHeaders(),
  })
  if (!response.ok) {
    throw new Error(`Backend unavailable: ${response.status}`)
  }
  return response.json()
}

export interface KBInfoResponse {
  index_name: string
  namespace: string
  embedding_model: string
  dimension: number
  dataset: string
  sample_size: number
  primary_intent: string
  retriever_k: number
  status: string
}

export interface KBSearchResult {
  content: string
  metadata: Record<string, any>
}

export async function fetchKBInfo(): Promise<KBInfoResponse> {
  const res = await fetch(`${API_BASE}/kb/info`, {
    headers: getHeaders(),
  })
  if (!res.ok) throw new Error("Failed to load KB info")
  return res.json()
}

export async function searchKB(query: string): Promise<{ query: string; results: KBSearchResult[] }> {
  const res = await fetch(`${API_BASE}/kb/search`, {
    method: "POST",
    headers: getHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ query }),
  })
  if (!res.ok) throw new Error("Failed to search KB")
  return res.json()
}

