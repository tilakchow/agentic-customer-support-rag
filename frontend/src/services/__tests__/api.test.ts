import { describe, it, expect, vi, beforeEach } from 'vitest'
import { streamMessage, checkHealth, fetchKBInfo, searchKB } from '../api'

describe('Frontend API Service', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('streamMessage throws on API error status', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 401,
        statusText: 'Unauthorized',
        json: async () => ({ detail: 'API Key missing or invalid' }),
      })
    )

    await expect(
      streamMessage({ message: 'Hi' }, {})
    ).rejects.toThrow('API Key missing or invalid')
  })

  it('checkHealth calls /health and returns data', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ status: 'online', service: 'agentic-rag' }),
      })
    )

    const res = await checkHealth()
    expect(res.status).toBe('online')
    expect(res.service).toBe('agentic-rag')
  })

  it('fetchKBInfo calls /kb/info and returns details', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ index_name: 'test-idx', status: 'ready' }),
      })
    )

    const res = await fetchKBInfo()
    expect(res.index_name).toBe('test-idx')
  })

  it('searchKB posts query to /kb/search', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ query: 'return policy', results: [] }),
      })
    )

    const res = await searchKB('return policy')
    expect(res.query).toBe('return policy')
    expect(res.results).toEqual([])
  })

  it('streamMessage processes SSE stream events correctly', async () => {
    const sseChunks = [
      'data: {"type": "status", "message": "Analyzing question...", "node": "route_question"}\n\n',
      'data: {"type": "token", "token": "Hello "}\n\n',
      'data: {"type": "token", "token": "there!"}\n\n',
      'data: {"type": "done", "thread_id": "thread-123", "source_used": "direct"}\n\n',
    ]

    const encoder = new TextEncoder()
    const stream = new ReadableStream({
      start(controller) {
        for (const chunk of sseChunks) {
          controller.enqueue(encoder.encode(chunk))
        }
        controller.close()
      },
    })

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        body: stream,
      })
    )

    const onStatus = vi.fn()
    const onToken = vi.fn()
    const onDone = vi.fn()
    const onError = vi.fn()

    await streamMessage(
      { message: 'Hello' },
      { onStatus, onToken, onDone, onError }
    )

    expect(onStatus).toHaveBeenCalledWith('Analyzing question...', 'route_question')
    expect(onToken).toHaveBeenCalledWith('Hello ')
    expect(onToken).toHaveBeenCalledWith('there!')
    expect(onDone).toHaveBeenCalledWith({
      thread_id: 'thread-123',
      source_used: 'direct',
      full_content: undefined,
    })
    expect(onError).not.toHaveBeenCalled()
  })
})
