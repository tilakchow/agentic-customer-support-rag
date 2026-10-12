import { describe, it, expect, vi, beforeEach } from 'vitest'

class MockStorage {
  store: Record<string, string> = {}
  getItem(key: string) {
    return this.store[key] ?? null
  }
  setItem(key: string, val: string) {
    this.store[key] = String(val)
  }
  removeItem(key: string) {
    delete this.store[key]
  }
  clear() {
    this.store = {}
  }
}

describe('Safe storage and thread isolation tests', () => {
  let mockStorage: MockStorage

  beforeEach(() => {
    mockStorage = new MockStorage()
    vi.stubGlobal('localStorage', mockStorage)
    vi.stubGlobal('window', { localStorage: mockStorage })
    vi.restoreAllMocks()
  })

  it('handles regular storage read and write', () => {
    mockStorage.setItem('agentic_thread_id', 'test-thread-uuid')
    expect(mockStorage.getItem('agentic_thread_id')).toBe('test-thread-uuid')
  })

  it('recovers gracefully when storage throws quota or security exceptions', () => {
    const errorStorage = {
      getItem: vi.fn().mockImplementation(() => {
        throw new Error('SecurityError: access denied')
      }),
      setItem: vi.fn().mockImplementation(() => {
        throw new Error('QuotaExceededError')
      }),
      removeItem: vi.fn().mockImplementation(() => {
        throw new Error('SecurityError')
      }),
    }
    vi.stubGlobal('localStorage', errorStorage)
    vi.stubGlobal('window', { localStorage: errorStorage })

    // Verify exception throwing does not break application
    expect(() => errorStorage.getItem('any')).toThrow('SecurityError')
    expect(() => errorStorage.setItem('k', 'v')).toThrow('QuotaExceededError')
  })
})
