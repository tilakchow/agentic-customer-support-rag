import { useState, useEffect, useCallback } from "react"
import { SavedThread } from "../components/views/HistoryView"
import { MessageItem } from "../components/ChatMessage"

const STORAGE_KEY_THREAD_ID = "agentic_thread_id"
const STORAGE_KEY_SAVED_THREADS = "agentic_saved_threads"

function safeGetItem(key: string): string | null {
  try {
    return typeof window !== "undefined" && window.localStorage
      ? window.localStorage.getItem(key)
      : null
  } catch (e) {
    console.warn(`localStorage read failed for "${key}":`, e)
    return null
  }
}

function safeSetItem(key: string, value: string): void {
  try {
    if (typeof window !== "undefined" && window.localStorage) {
      window.localStorage.setItem(key, value)
    }
  } catch (e) {
    console.warn(`localStorage write failed for "${key}":`, e)
  }
}

function safeRemoveItem(key: string): void {
  try {
    if (typeof window !== "undefined" && window.localStorage) {
      window.localStorage.removeItem(key)
    }
  } catch (e) {
    console.warn(`localStorage remove failed for "${key}":`, e)
  }
}

export function useThreads() {
  const [threadId, setThreadId] = useState<string>(() => {
    return safeGetItem(STORAGE_KEY_THREAD_ID) || crypto.randomUUID()
  })

  const [savedThreads, setSavedThreads] = useState<SavedThread[]>(() => {
    const raw = safeGetItem(STORAGE_KEY_SAVED_THREADS)
    if (!raw) return []
    try {
      return JSON.parse(raw)
    } catch {
      return []
    }
  })

  // Sync threadId to storage
  useEffect(() => {
    safeSetItem(STORAGE_KEY_THREAD_ID, threadId)
  }, [threadId])

  // Sync savedThreads to storage
  useEffect(() => {
    safeSetItem(STORAGE_KEY_SAVED_THREADS, JSON.stringify(savedThreads))
  }, [savedThreads])

  const createNewThread = useCallback(() => {
    const newId = crypto.randomUUID()
    setThreadId(newId)
    return newId
  }, [])

  const resumeThread = useCallback((thread: SavedThread) => {
    setThreadId(thread.threadId)
  }, [])

  const deleteThread = useCallback(
    (targetId: string) => {
      setSavedThreads((prev) => prev.filter((t) => t.threadId !== targetId))
      if (targetId === threadId) {
        createNewThread()
      }
    },
    [threadId, createNewThread]
  )

  const clearAllThreads = useCallback(() => {
    setSavedThreads([])
    createNewThread()
  }, [createNewThread])

  const clearStorage = useCallback(() => {
    safeRemoveItem(STORAGE_KEY_SAVED_THREADS)
    safeRemoveItem(STORAGE_KEY_THREAD_ID)
    setSavedThreads([])
    createNewThread()
  }, [createNewThread])

  const recordMessages = useCallback(
    (activeId: string, messages: MessageItem[]) => {
      if (messages.length === 0) return

      const firstUserMsg =
        messages.find((m) => m.role === "user")?.content || "Customer Inquiry"
      const previewText = messages[messages.length - 1].content.slice(0, 90)
      const titleText =
        firstUserMsg.length > 45 ? `${firstUserMsg.slice(0, 45)}...` : firstUserMsg

      setSavedThreads((prev) => {
        const existingIdx = prev.findIndex((t) => t.threadId === activeId)
        const updatedItem: SavedThread = {
          threadId: activeId,
          title: titleText,
          preview: previewText,
          updatedAt: new Date().toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          }),
          messageCount: messages.length,
          messages,
        }

        if (existingIdx >= 0) {
          const copy = [...prev]
          copy[existingIdx] = updatedItem
          return copy
        } else {
          return [updatedItem, ...prev]
        }
      })
    },
    []
  )

  return {
    threadId,
    setThreadId,
    savedThreads,
    setSavedThreads,
    createNewThread,
    resumeThread,
    deleteThread,
    clearAllThreads,
    clearStorage,
    recordMessages,
  }
}
