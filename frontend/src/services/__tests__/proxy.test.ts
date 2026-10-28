import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { checkEdgeRateLimit, isAllowedOrigin, getClientIp } from '../../../api/_proxy'

describe('Edge Proxy Security & Rate Limiting Helpers', () => {
  beforeEach(() => {
    // Each test uses isolated IPs to avoid interference
  })

  describe('checkEdgeRateLimit', () => {
    it('allows requests within threshold', () => {
      const ip = '10.0.0.1'
      expect(checkEdgeRateLimit(ip, 5, 10000)).toBe(true)
      expect(checkEdgeRateLimit(ip, 5, 10000)).toBe(true)
      expect(checkEdgeRateLimit(ip, 5, 10000)).toBe(true)
    })

    it('blocks requests once threshold is exceeded', () => {
      const ip = '10.0.0.2'
      for (let i = 0; i < 3; i++) {
        expect(checkEdgeRateLimit(ip, 3, 10000)).toBe(true)
      }
      // 4th request must be blocked
      expect(checkEdgeRateLimit(ip, 3, 10000)).toBe(false)
    })
  })

  describe('isAllowedOrigin', () => {
    it('allows requests with no origin header (same-origin / server-to-server)', () => {
      const req = new Request('https://agentic-customer-support-rag.vercel.app/api/chat')
      expect(isAllowedOrigin(req)).toBe(true)
    })

    it('allows trusted frontend origins', () => {
      const req = new Request('https://agentic-customer-support-rag.vercel.app/api/chat', {
        headers: { origin: 'https://agentic-customer-support-rag.vercel.app' },
      })
      expect(isAllowedOrigin(req)).toBe(true)
    })

    it('allows local dev origins', () => {
      const req = new Request('http://localhost:5173/api/chat', {
        headers: { origin: 'http://localhost:5173' },
      })
      expect(isAllowedOrigin(req)).toBe(true)
    })

    it('rejects untrusted third-party origins', () => {
      const req = new Request('https://agentic-customer-support-rag.vercel.app/api/chat', {
        headers: { origin: 'https://malicious-phishing-site.com' },
      })
      expect(isAllowedOrigin(req)).toBe(false)
    })
  })

  describe('getClientIp', () => {
    it('extracts client IP from x-forwarded-for header', () => {
      const req = new Request('https://example.com/api/chat', {
        headers: { 'x-forwarded-for': '203.0.113.195, 70.41.3.18' },
      })
      expect(getClientIp(req)).toBe('203.0.113.195')
    })

    it('falls back to x-real-ip when x-forwarded-for is missing', () => {
      const req = new Request('https://example.com/api/chat', {
        headers: { 'x-real-ip': '198.51.100.42' },
      })
      expect(getClientIp(req)).toBe('198.51.100.42')
    })

    it('defaults to 127.0.0.1 when no headers present', () => {
      const req = new Request('https://example.com/api/chat')
      expect(getClientIp(req)).toBe('127.0.0.1')
    })
  })

  describe('API Route Handlers Method & Configuration Verification', () => {
    it('verifies chat handler enforces POST method', async () => {
      const { default: chatHandler } = await import('../../../api/chat.js')
      const getReq = new Request('https://example.com/api/chat', { method: 'GET' })
      const res = await chatHandler(getReq)
      expect(res.status).toBe(405)
      const data = await res.json()
      expect(data.error).toBe('Method not allowed')
    })

    it('verifies chat-stream handler enforces POST method', async () => {
      const { default: streamHandler } = await import('../../../api/chat-stream.js')
      const getReq = new Request('https://example.com/api/chat-stream', { method: 'GET' })
      const res = await streamHandler(getReq)
      expect(res.status).toBe(405)
      const data = await res.json()
      expect(data.error).toBe('Method not allowed')
    })

    it('verifies kb/search handler enforces POST method', async () => {
      const { default: searchHandler } = await import('../../../api/kb/search.js')
      const getReq = new Request('https://example.com/api/kb/search', { method: 'GET' })
      const res = await searchHandler(getReq)
      expect(res.status).toBe(405)
      const data = await res.json()
      expect(data.error).toBe('Method not allowed')
    })

    it('verifies health and kb/info handlers are functions and have edge config', async () => {
      const { default: healthHandler, config: healthConfig } = await import('../../../api/health.js')
      const { default: infoHandler, config: infoConfig } = await import('../../../api/kb/info.js')
      expect(typeof healthHandler).toBe('function')
      expect(healthConfig.runtime).toBe('edge')
      expect(typeof infoHandler).toBe('function')
      expect(infoConfig.runtime).toBe('edge')
    })
  })

  describe('Endpoint Proxy Execution, Streaming, Error Handling & Secret Protection', () => {
    const originalFetch = globalThis.fetch

    afterEach(() => {
      globalThis.fetch = originalFetch
    })

    it('health handler proxies GET /health successfully', async () => {
      const { default: healthHandler } = await import('../../../api/health.js')
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ status: 'ok', environment: 'production' }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      )

      const req = new Request('https://agentic-customer-support-rag.vercel.app/api/health')
      const res = await healthHandler(req)
      expect(res.status).toBe(200)
      const data = await res.json()
      expect(data.status).toBe('ok')
      expect(globalThis.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/health'),
        expect.objectContaining({ method: 'GET' })
      )
    })

    it('kb/info handler proxies GET /kb/info successfully', async () => {
      const { default: infoHandler } = await import('../../../api/kb/info.js')
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ status: 'ready', total_chunks: 100 }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      )

      const req = new Request('https://agentic-customer-support-rag.vercel.app/api/kb/info')
      const res = await infoHandler(req)
      expect(res.status).toBe(200)
      const data = await res.json()
      expect(data.status).toBe('ready')
    })

    it('kb/search handler proxies POST /kb/search with query payload', async () => {
      const { default: searchHandler } = await import('../../../api/kb/search.js')
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ results: [{ title: 'Doc 1', content: 'Info' }] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      )

      const req = new Request('https://agentic-customer-support-rag.vercel.app/api/kb/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: 'return policy', top_k: 3 }),
      })
      const res = await searchHandler(req)
      expect(res.status).toBe(200)
      const data = await res.json()
      expect(data.results).toHaveLength(1)
    })

    it('chat handler proxies POST /chat with message payload', async () => {
      const { default: chatHandler } = await import('../../../api/chat.js')
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ answer: 'Here is your answer', route: 'kb' }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      )

      const req = new Request('https://agentic-customer-support-rag.vercel.app/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: 'Hello!' }),
      })
      const res = await chatHandler(req)
      expect(res.status).toBe(200)
      const data = await res.json()
      expect(data.answer).toBe('Here is your answer')
    })

    it('chat-stream handler proxies POST /chat/stream with event-stream headers', async () => {
      const { default: streamHandler } = await import('../../../api/chat-stream.js')
      const streamText = 'data: {"event": "status", "node": "classify_scope"}\n\n'
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(streamText, {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        })
      )

      const req = new Request('https://agentic-customer-support-rag.vercel.app/api/chat-stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: 'Streaming query' }),
      })
      const res = await streamHandler(req)
      expect(res.status).toBe(200)
      expect(res.headers.get('Content-Type')).toBe('text/event-stream')
      expect(res.headers.get('Cache-Control')).toContain('no-cache')
    })

    it('handles backend connection failure by returning clean 502 without leaking secrets or traces', async () => {
      const { default: healthHandler } = await import('../../../api/health.js')
      globalThis.fetch = vi.fn().mockRejectedValue(new Error('ECONNREFUSED 127.0.0.1:8000'))

      const req = new Request('https://agentic-customer-support-rag.vercel.app/api/health')
      const res = await healthHandler(req)
      expect(res.status).toBe(502)
      const data = await res.json()
      expect(data.error).toBe('Backend service is currently unavailable. Please try again shortly.')
      // Ensure no raw internal error details leaked
      expect(JSON.stringify(data)).not.toContain('ECONNREFUSED')
      expect(JSON.stringify(data)).not.toContain('API_KEY')
    })
  })
})

