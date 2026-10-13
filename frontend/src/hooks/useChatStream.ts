import { useState, useRef, useEffect, useCallback } from "react"
import { streamMessage, StreamCallbacks } from "../services/api"

export interface UseChatStreamOptions {
  threadId: string
  onToken?: (token: string) => void
  onComplete?: (data: { thread_id: string; source_used?: string; full_content?: string }) => void
  onError?: (error: string) => void
}

export function useChatStream(options?: UseChatStreamOptions) {
  const [loading, setLoading] = useState(false)
  const [statusMessage, setStatusMessage] = useState("")
  const [activeNode, setActiveNode] = useState<string | undefined>(undefined)
  const [visitedNodes, setVisitedNodes] = useState<string[]>([])
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  const activeThreadIdRef = useRef<string | undefined>(options?.threadId)
  const isStreamingRef = useRef(false)
  const abortControllerRef = useRef<AbortController | null>(null)

  const abortStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    isStreamingRef.current = false
    setLoading(false)
    setStatusMessage("")
    setActiveNode(undefined)
  }, [])

  // Abort in-flight stream if threadId changes
  useEffect(() => {
    if (options?.threadId && options.threadId !== activeThreadIdRef.current) {
      if (isStreamingRef.current) {
        console.warn(`Thread switched from ${activeThreadIdRef.current} to ${options.threadId}; aborting active stream.`)
        abortStream()
      }
      activeThreadIdRef.current = options.threadId
    }
  }, [options?.threadId, abortStream])

  const startStream = useCallback(
    async (
      query: string,
      targetThreadId: string,
      callbacks: StreamCallbacks
    ) => {
      // Ensure previous stream is cleanly stopped
      abortStream()

      const currentStreamThreadId = targetThreadId
      const controller = new AbortController()
      abortControllerRef.current = controller

      isStreamingRef.current = true
      setLoading(true)
      setErrorMsg(null)
      setStatusMessage("Routing question...")
      setActiveNode("route_question")
      setVisitedNodes(["route_question"])

      try {
        await streamMessage(
          {
            message: query,
            thread_id: targetThreadId,
          },
          {
            onStatus: (status, node) => {
              if (!isStreamingRef.current || activeThreadIdRef.current !== currentStreamThreadId) {
                return
              }
              setStatusMessage(status)
              if (node) {
                setActiveNode(node)
                setVisitedNodes((prev) => Array.from(new Set([...prev, node])))
              }
              callbacks.onStatus?.(status, node)
            },
            onToken: (token) => {
              if (!isStreamingRef.current || activeThreadIdRef.current !== currentStreamThreadId) {
                return
              }
              callbacks.onToken?.(token)
              options?.onToken?.(token)
            },
            onDone: (data) => {
              if (!isStreamingRef.current || activeThreadIdRef.current !== currentStreamThreadId) {
                return
              }
              callbacks.onDone?.(data)
              options?.onComplete?.(data)
            },
            onError: (err) => {
              if (!isStreamingRef.current || activeThreadIdRef.current !== currentStreamThreadId) {
                return
              }
              setErrorMsg(err)
              callbacks.onError?.(err)
              options?.onError?.(err)
            },
          },
          controller.signal
        )
      } catch (err: any) {
        if (controller.signal.aborted) {
          return // User initiated abort, do not surface as error
        }
        if (isStreamingRef.current && activeThreadIdRef.current === currentStreamThreadId) {
          const message = err?.message || "Failed to communicate with agent service."
          setErrorMsg(message)
          callbacks.onError?.(message)
          options?.onError?.(message)
        }
      } finally {
        // Guaranteed cleanup: loading must never stay stuck true
        if (activeThreadIdRef.current === currentStreamThreadId) {
          setLoading(false)
          setStatusMessage("")
          setActiveNode(undefined)
          isStreamingRef.current = false
          if (abortControllerRef.current === controller) {
            abortControllerRef.current = null
          }
        }
      }
    },
    [abortStream, options]
  )

  return {
    loading,
    statusMessage,
    activeNode,
    visitedNodes,
    errorMsg,
    setErrorMsg,
    setActiveNode,
    setVisitedNodes,
    setStatusMessage,
    startStream,
    abortStream,
  }
}
