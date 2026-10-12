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

  // Abort in-flight stream if threadId changes
  useEffect(() => {
    if (options?.threadId && options.threadId !== activeThreadIdRef.current) {
      if (isStreamingRef.current) {
        console.warn(`Thread switched from ${activeThreadIdRef.current} to ${options.threadId}; aborting active stream.`)
        isStreamingRef.current = false
        setLoading(false)
        setStatusMessage("")
        setActiveNode(undefined)
      }
      activeThreadIdRef.current = options.threadId
    }
  }, [options?.threadId])

  const abortStream = useCallback(() => {
    isStreamingRef.current = false
    setLoading(false)
    setStatusMessage("")
    setActiveNode(undefined)
  }, [])

  const startStream = useCallback(
    async (
      query: string,
      targetThreadId: string,
      callbacks: StreamCallbacks
    ) => {
      // Ensure previous stream is stopped
      abortStream()

      const currentStreamThreadId = targetThreadId
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
              // Ignore tokens/events if thread switched during fetch
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
              setLoading(false)
              setStatusMessage("")
              setActiveNode(undefined)
              isStreamingRef.current = false
              callbacks.onDone?.(data)
              options?.onComplete?.(data)
            },
            onError: (err) => {
              if (!isStreamingRef.current || activeThreadIdRef.current !== currentStreamThreadId) {
                return
              }
              setLoading(false)
              setErrorMsg(err)
              setStatusMessage("")
              setActiveNode(undefined)
              isStreamingRef.current = false
              callbacks.onError?.(err)
              options?.onError?.(err)
            },
          }
        )
      } catch (err: any) {
        if (isStreamingRef.current && activeThreadIdRef.current === currentStreamThreadId) {
          const message = err?.message || "Failed to communicate with agent service."
          setErrorMsg(message)
          setLoading(false)
          setStatusMessage("")
          setActiveNode(undefined)
          isStreamingRef.current = false
          callbacks.onError?.(message)
          options?.onError?.(message)
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
