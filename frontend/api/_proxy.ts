/// <reference types="node" />

// In-memory sliding rate limit tracker per client IP at the edge
const ipRequestHistory = new Map<string, { count: number; resetAt: number }>()

export function checkEdgeRateLimit(
  clientIp: string,
  maxRequests: number = 30,
  windowMs: number = 60000
): boolean {
  const now = Date.now()
  const entry = ipRequestHistory.get(clientIp)

  if (!entry || now > entry.resetAt) {
    ipRequestHistory.set(clientIp, { count: 1, resetAt: now + windowMs })
    return true
  }

  if (entry.count >= maxRequests) {
    return false
  }

  entry.count += 1
  return true
}

export function isAllowedOrigin(req: Request): boolean {
  const origin = req.headers.get("origin")
  if (!origin) return true // Same-origin or non-browser request

  const allowedOrigins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://agentic-customer-support-rag.vercel.app",
  ]
  const customOrigin = process.env.ALLOWED_ORIGIN
  if (customOrigin && !allowedOrigins.includes(customOrigin)) {
    allowedOrigins.push(customOrigin)
  }

  return allowedOrigins.some((allowed) => origin.startsWith(allowed))
}

export function getClientIp(req: Request): string {
  const xff = req.headers.get("x-forwarded-for")
  if (xff) {
    return xff.split(",")[0].trim()
  }
  return req.headers.get("x-real-ip") || "127.0.0.1"
}

export async function proxyToBackend(
  req: Request,
  path: string,
  isStreaming = false
): Promise<Response> {
  if (!isAllowedOrigin(req)) {
    return new Response(JSON.stringify({ error: "Access denied: Unauthorized origin" }), {
      status: 403,
      headers: { "Content-Type": "application/json" },
    })
  }

  const clientIp = getClientIp(req)
  if (!checkEdgeRateLimit(clientIp)) {
    return new Response(
      JSON.stringify({ error: "Too many requests. Please slow down and try again later." }),
      {
        status: 429,
        headers: { "Content-Type": "application/json", "Retry-After": "60" },
      }
    )
  }

  const backendUrl =
    process.env.BACKEND_URL || process.env.VITE_API_URL || "http://localhost:8000"
  const targetUrl = `${backendUrl.replace(/\/$/, "")}${path}`

  try {
    const apiKey = process.env.API_KEY || req.headers.get("x-api-key") || ""
    const sessionId = req.headers.get("x-session-id") || req.headers.get("x-client-session") || ""
    const headers: Record<string, string> = {
      ...(apiKey ? { "X-API-Key": apiKey } : {}),
      ...(sessionId ? { "X-Session-ID": sessionId } : {}),
      "X-Forwarded-For": clientIp,
    }

    const contentType = req.headers.get("content-type")
    if (contentType) {
      headers["Content-Type"] = contentType
    }

    const body =
      req.method === "POST" || req.method === "PUT" || req.method === "PATCH"
        ? await req.text()
        : undefined

    const response = await fetch(targetUrl, {
      method: req.method,
      headers,
      body,
    })

    if (isStreaming) {
      return new Response(response.body, {
        status: response.status,
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache, no-transform",
          Connection: "keep-alive",
        },
      })
    }

    return new Response(response.body, {
      status: response.status,
      headers: {
        "Content-Type": response.headers.get("content-type") || "application/json",
      },
    })
  } catch (error) {
    console.error(`[Edge Proxy Error] Failed to proxy request to ${targetUrl}:`, error)
    return new Response(
      JSON.stringify({ error: "Backend service is currently unavailable. Please try again shortly." }),
      {
        status: 502,
        headers: { "Content-Type": "application/json" },
      }
    )
  }
}
