import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useThreads } from '../useThreads'
import { useChatStream } from '../useChatStream'
import * as apiModule from '../../services/api'

describe('useThreads hook', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.restoreAllMocks()
  })

  it('initializes with a valid threadId and persists to localStorage', () => {
    const { result } = renderHook(() => useThreads())
    expect(result.current.threadId).toBeDefined()
    expect(typeof result.current.threadId).toBe('string')
    expect(result.current.threadId.length).toBeGreaterThan(0)
    expect(localStorage.getItem('agentic_thread_id')).toBe(result.current.threadId)
  })

  it('restores threadId and saved threads from localStorage if present', () => {
    const existingThread = {
      threadId: 'stored-thread-1',
      title: 'Saved Conversation',
      preview: 'Hello world',
      updatedAt: '10:00 AM',
      messageCount: 2,
      messages: [],
    }
    localStorage.setItem('agentic_thread_id', 'stored-thread-1')
    localStorage.setItem('agentic_saved_threads', JSON.stringify([existingThread]))

    const { result } = renderHook(() => useThreads())
    expect(result.current.threadId).toBe('stored-thread-1')
    expect(result.current.savedThreads).toHaveLength(1)
    expect(result.current.savedThreads[0].title).toBe('Saved Conversation')
  })

  it('createNewThread generates a new thread id', () => {
    const { result } = renderHook(() => useThreads())
    const initialId = result.current.threadId

    let newId = ''
    act(() => {
      newId = result.current.createNewThread()
    })

    expect(newId).not.toBe(initialId)
    expect(result.current.threadId).toBe(newId)
  })

  it('resumeThread switches active thread id', () => {
    const { result } = renderHook(() => useThreads())
    act(() => {
      result.current.resumeThread({
        threadId: 'thread-resume-99',
        title: 'Thread 99',
        preview: 'Preview',
        updatedAt: '11:00 AM',
        messageCount: 1,
        messages: [],
      })
    })

    expect(result.current.threadId).toBe('thread-resume-99')
  })

  it('deleteThread removes thread and regenerates active thread if current is deleted', () => {
    const { result } = renderHook(() => useThreads())
    const currentId = result.current.threadId

    act(() => {
      result.current.setSavedThreads([
        {
          threadId: currentId,
          title: 'Current Thread',
          preview: 'Preview',
          updatedAt: '12:00 PM',
          messageCount: 1,
          messages: [],
        },
      ])
    })

    act(() => {
      result.current.deleteThread(currentId)
    })

    expect(result.current.savedThreads).toHaveLength(0)
    expect(result.current.threadId).not.toBe(currentId)
  })

  it('recordMessages creates or updates thread history', () => {
    const { result } = renderHook(() => useThreads())
    const activeId = result.current.threadId

    act(() => {
      result.current.recordMessages(activeId, [
        { id: '1', role: 'user', content: 'What is your refund policy?', timestamp: '10:00 AM' },
        { id: '2', role: 'assistant', content: 'Refunds are allowed within 30 days.', timestamp: '10:01 AM' },
      ])
    })

    expect(result.current.savedThreads).toHaveLength(1)
    expect(result.current.savedThreads[0].title).toBe('What is your refund policy?')
    expect(result.current.savedThreads[0].preview).toContain('Refunds are allowed')
    expect(result.current.savedThreads[0].messageCount).toBe(2)
  })

  it('clearStorage wipes storage and starts a fresh thread', () => {
    const { result } = renderHook(() => useThreads())
    act(() => {
      result.current.clearStorage()
    })

    expect(result.current.savedThreads).toEqual([])
    expect(result.current.threadId).toBeDefined()
  })
})

describe('useChatStream hook', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('initializes with default idle state', () => {
    const { result } = renderHook(() => useChatStream({ threadId: 'thread-1' }))
    expect(result.current.loading).toBe(false)
    expect(result.current.statusMessage).toBe('')
    expect(result.current.activeNode).toBeUndefined()
    expect(result.current.visitedNodes).toEqual([])
    expect(result.current.errorMsg).toBeNull()
  })

  it('abortStream resets in-flight streaming state', () => {
    const { result } = renderHook(() => useChatStream({ threadId: 'thread-1' }))
    act(() => {
      result.current.setStatusMessage('Routing...')
      result.current.setActiveNode('route_question')
      result.current.abortStream()
    })

    expect(result.current.loading).toBe(false)
    expect(result.current.statusMessage).toBe('')
    expect(result.current.activeNode).toBeUndefined()
  })

  it('startStream manages stream lifecycle through status, token, and completion', async () => {
    vi.spyOn(apiModule, 'streamMessage').mockImplementation(async (_req, callbacks) => {
      callbacks.onStatus?.('Searching knowledge base...', 'retrieve_kb')
      callbacks.onToken?.('Answer chunk 1')
      callbacks.onToken?.(' chunk 2')
      callbacks.onDone?.({ thread_id: 'thread-1', source_used: 'private_kb' })
    })

    const onToken = vi.fn()
    const onComplete = vi.fn()
    const { result } = renderHook(() =>
      useChatStream({ threadId: 'thread-1', onToken, onComplete })
    )

    await act(async () => {
      await result.current.startStream('My question', 'thread-1', {})
    })

    expect(apiModule.streamMessage).toHaveBeenCalledTimes(1)
    expect(onToken).toHaveBeenCalledWith('Answer chunk 1')
    expect(onToken).toHaveBeenCalledWith(' chunk 2')
    expect(onComplete).toHaveBeenCalledWith({
      thread_id: 'thread-1',
      source_used: 'private_kb',
    })
    expect(result.current.loading).toBe(false)
  })

  it('guards against stale thread tokens when active thread changes', async () => {
    let savedCallbacks: any = null
    vi.spyOn(apiModule, 'streamMessage').mockImplementation(async (_req, callbacks) => {
      savedCallbacks = callbacks
    })

    const onToken = vi.fn()
    const { result, rerender } = renderHook(
      ({ threadId }) => useChatStream({ threadId, onToken }),
      { initialProps: { threadId: 'thread-A' } }
    )

    await act(async () => {
      // Start stream for thread A
      result.current.startStream('Hello', 'thread-A', {})
    })

    // User switches to thread B
    rerender({ threadId: 'thread-B' })

    // Late arriving token for thread A
    act(() => {
      savedCallbacks.onToken('Late token from thread A')
    })

    // Token must be ignored by stale-thread guard
    expect(onToken).not.toHaveBeenCalledWith('Late token from thread A')
  })

  it('handles stream errors gracefully', async () => {
    vi.spyOn(apiModule, 'streamMessage').mockImplementation(async (_req, callbacks) => {
      callbacks.onError?.('Network error')
    })

    const onError = vi.fn()
    const { result } = renderHook(() =>
      useChatStream({ threadId: 'thread-1', onError })
    )

    await act(async () => {
      await result.current.startStream('Question', 'thread-1', {})
    })

    expect(onError).toHaveBeenCalledWith('Network error')
    expect(result.current.errorMsg).toBe('Network error')
    expect(result.current.loading).toBe(false)
  })
})
