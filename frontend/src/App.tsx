import React, { useState, useEffect, useRef } from "react"
import { AppSidebar } from "./components/AppSidebar"
import { WelcomeHero } from "./components/WelcomeHero"
import { ChatMessage, MessageItem } from "./components/ChatMessage"
import { Composer } from "./components/Composer"
import { StreamingTimeline } from "./components/StreamingTimeline"
import { ErrorBanner } from "./components/ErrorBanner"
import { Dialog } from "./components/ui/Dialog"
import { KBView } from "./components/views/KBView"
import { SettingsView } from "./components/views/SettingsView"
import { SavedThread } from "./components/views/HistoryView"
import { ErrorBoundary } from "./components/ErrorBoundary"
import { useThreads } from "./hooks/useThreads"
import { useChatStream } from "./hooks/useChatStream"
import { checkHealth } from "./services/api"
import { Search, Copy, Check, PanelLeft } from "lucide-react"

export function AppContent() {
  const [messages, setMessages] = useState<MessageItem[]>([])
  const [input, setInput] = useState("")
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [kbDialogOpen, setKbDialogOpen] = useState(false)
  const [settingsDialogOpen, setSettingsDialogOpen] = useState(false)
  const [searchFilter, setSearchFilter] = useState("")
  const [showSearch, setShowSearch] = useState(false)
  const [backendOnline, setBackendOnline] = useState(false)
  const [copiedThread, setCopiedThread] = useState(false)

  // Theme setup: defaults to system preference, toggleable
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    try {
      const stored = localStorage.getItem("bitext_theme")
      if (stored === "dark" || stored === "light") return stored
      return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"
    } catch {
      return "light"
    }
  })

  useEffect(() => {
    try {
      localStorage.setItem("bitext_theme", theme)
    } catch {}
    if (theme === "dark") {
      document.documentElement.classList.add("dark")
    } else {
      document.documentElement.classList.remove("dark")
    }
  }, [theme])

  const toggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"))
  }

  const {
    threadId,
    setThreadId,
    savedThreads,
    createNewThread,
    resumeThread,
    deleteThread,
    clearStorage,
    recordMessages,
  } = useThreads()

  const {
    loading,
    statusMessage,
    activeNode,
    visitedNodes,
    errorMsg,
    setErrorMsg,
    startStream,
    abortStream,
  } = useChatStream({ threadId })

  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Sync messages with saved threads in storage
  useEffect(() => {
    recordMessages(threadId, messages)
  }, [messages, threadId, recordMessages])

  // Resilient, respectful health checking:
  // - Performs initial verification on mount
  // - Re-verifies when tab regains visibility/focus
  // - Polls with 45s interval, avoiding concurrent runs and tight retry loops
  // - Pauses polling when the browser tab is hidden to save client & server resources
  useEffect(() => {
    let isMounted = true
    let timerId: ReturnType<typeof setTimeout> | null = null

    const verify = async () => {
      if (typeof document !== "undefined" && document.hidden) return

      try {
        await checkHealth()
        if (isMounted) setBackendOnline(true)
      } catch {
        if (isMounted) setBackendOnline(false)
      } finally {
        if (isMounted) {
          timerId = setTimeout(verify, 45000)
        }
      }
    }

    verify()

    const handleVisibilityChange = () => {
      if (typeof document !== "undefined" && !document.hidden) {
        if (timerId) clearTimeout(timerId)
        verify()
      }
    }

    if (typeof document !== "undefined") {
      document.addEventListener("visibilitychange", handleVisibilityChange)
    }

    return () => {
      isMounted = false
      if (timerId) clearTimeout(timerId)
      if (typeof document !== "undefined") {
        document.removeEventListener("visibilitychange", handleVisibilityChange)
      }
    }
  }, [])

  // Auto scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, loading, statusMessage])

  // New Chat action
  const handleNewChat = () => {
    abortStream()
    createNewThread()
    setMessages([])
    setErrorMsg(null)
    setInput("")
    setShowSearch(false)
    setSearchFilter("")
  }

  // Resume thread
  const handleResumeThread = (thread: SavedThread) => {
    abortStream()
    resumeThread(thread)
    setMessages(thread.messages)
  }

  // Send message
  const handleSend = async (textToSend?: string) => {
    const query = (textToSend || input).trim()
    if (!query || loading) return

    setErrorMsg(null)
    const userTimestamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    const userMsg: MessageItem = {
      id: crypto.randomUUID(),
      role: "user",
      content: query,
      timestamp: userTimestamp,
    }

    const assistantId = crypto.randomUUID()
    const initialAssistantMsg: MessageItem = {
      id: assistantId,
      role: "assistant",
      content: "",
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    }

    setMessages((prev) => [...prev, userMsg, initialAssistantMsg])
    setInput("")

    await startStream(query, threadId, {
      onToken: (token) => {
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + token } : m))
        )
      },
      onDone: (data) => {
        setMessages((prev) =>
          prev.map((m) => {
            if (m.id === assistantId) {
              return {
                ...m,
                content: m.content || data.full_content || "",
                source: data.source_used,
                sources: data.sources,
              }
            }
            return m
          })
        )
        if (data.thread_id && data.thread_id !== threadId) {
          setThreadId(data.thread_id)
        }
      },
      onError: (_err) => {
        setMessages((prev) => prev.filter((m) => m.id !== assistantId || m.content.length > 0))
      },
    })
  }

  // Regenerate last response
  const handleRegenerate = () => {
    const lastUserMsg = [...messages].reverse().find((m) => m.role === "user")
    if (lastUserMsg) {
      handleSend(lastUserMsg.content)
    }
  }

  const copyThreadId = () => {
    navigator.clipboard.writeText(threadId)
    setCopiedThread(true)
    setTimeout(() => setCopiedThread(false), 2000)
  }

  // Message search filter
  const displayedMessages = searchFilter
    ? messages.filter((m) => m.content.toLowerCase().includes(searchFilter.toLowerCase()))
    : messages

  return (
    <div className="flex h-screen w-screen bg-[var(--color-canvas)] text-[var(--color-ink)] overflow-hidden font-sans">
      {/* Perplexity-style Minimal Collapsible Sidebar */}
      <AppSidebar
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed((prev) => !prev)}
        currentThreadId={threadId}
        threads={savedThreads}
        onSelectThread={handleResumeThread}
        onDeleteThread={deleteThread}
        onNewChat={handleNewChat}
        onOpenKB={() => setKbDialogOpen(true)}
        onOpenSettings={() => setSettingsDialogOpen(true)}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      {/* Main Single Centered Reading Column Workspace */}
      <div className="flex-1 flex flex-col h-full overflow-hidden relative">
        {/* Subtle Top Header */}
        <header className="h-12 border-b border-[var(--color-border-subtle)] px-4 flex items-center justify-between bg-[var(--color-canvas)] select-none shrink-0 z-10">
          <div className="flex items-center gap-2">
            {sidebarCollapsed && (
              <button
                type="button"
                onClick={() => setSidebarCollapsed(false)}
                className="p-1.5 rounded-[var(--radius-sm)] text-[var(--color-ink-muted)] hover:text-[var(--color-ink)] hover:bg-[var(--color-surface-hover)] transition-colors cursor-pointer"
                aria-label="Expand navigation rail"
              >
                <PanelLeft className="w-4 h-4" />
              </button>
            )}
            <span className="text-[13px] font-semibold text-[var(--color-ink)]">
              Bitext Support
            </span>
            <span
              className={`w-2 h-2 rounded-full ${
                backendOnline ? "bg-[var(--color-success)]" : "bg-amber-500"
              }`}
              title={backendOnline ? "FastAPI Connected" : "FastAPI Offline"}
            />
          </div>

          <div className="flex items-center gap-2">
            {/* Search within conversation */}
            <button
              type="button"
              onClick={() => setShowSearch((prev) => !prev)}
              className="p-1.5 rounded-[var(--radius-sm)] text-[var(--color-ink-muted)] hover:text-[var(--color-ink)] hover:bg-[var(--color-surface-hover)] transition-colors cursor-pointer"
              aria-label="Search conversation messages"
              title="Search conversation"
            >
              <Search className="w-4 h-4" />
            </button>

            {/* Session Thread ID Pill */}
            <button
              type="button"
              onClick={copyThreadId}
              className="hidden sm:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-[var(--radius-full)] bg-[var(--color-surface)] border border-[var(--color-border)] text-[11px] font-mono text-[var(--color-ink-muted)] hover:border-[var(--color-brand)] transition-colors cursor-pointer"
              aria-label="Copy active thread identifier"
              title="Copy Thread ID"
            >
              <span className="text-[var(--color-ink-faint)]">thread:</span>
              <span>{threadId.slice(0, 8)}...</span>
              {copiedThread ? (
                <Check className="w-3 h-3 text-[var(--color-success)]" />
              ) : (
                <Copy className="w-3 h-3 text-[var(--color-ink-faint)]" />
              )}
            </button>
          </div>
        </header>

        {/* In-chat search filter bar */}
        {showSearch && (
          <div className="px-4 py-2 border-b border-[var(--color-border-subtle)] bg-[var(--color-surface)] flex items-center justify-between text-[12px]">
            <input
              type="text"
              value={searchFilter}
              onChange={(e) => setSearchFilter(e.target.value)}
              placeholder="Find keywords in this thread..."
              className="flex-1 max-w-sm px-2.5 py-1 rounded-[var(--radius-sm)] bg-[var(--color-canvas)] border border-[var(--color-border)] text-[12.5px] text-[var(--color-ink)] focus:outline-none focus:border-[var(--color-brand)]"
            />
            <button
              type="button"
              onClick={() => {
                setSearchFilter("")
                setShowSearch(false)
              }}
              className="text-[11.5px] text-[var(--color-ink-muted)] hover:text-[var(--color-ink)] cursor-pointer"
            >
              Close
            </button>
          </div>
        )}

        {/* Scrollable Center Reading Column (Capped at 840px max-width) */}
        <main className="flex-1 overflow-y-auto px-4 sm:px-6 py-6">
          <div className="max-w-[840px] mx-auto min-h-full flex flex-col justify-between">
            {/* Error Banner */}
            {errorMsg && (
              <ErrorBanner
                message={errorMsg}
                onRetry={handleRegenerate}
                onDismiss={() => setErrorMsg(null)}
              />
            )}

            {/* Empty State vs Chat Thread */}
            {messages.length === 0 ? (
              <div className="my-auto">
                <WelcomeHero onSelectQuery={(q) => handleSend(q)} />
              </div>
            ) : (
              <div className="flex-1 flex flex-col pb-6">
                {displayedMessages.map((msg, idx) => {
                  if (msg.role === "assistant" && !msg.content && loading) {
                    return null
                  }
                  const isLastAssistant =
                    msg.role === "assistant" && idx === displayedMessages.length - 1
                  return (
                    <ChatMessage
                      key={msg.id}
                      message={msg}
                      isLastAssistant={isLastAssistant}
                      onRegenerate={handleRegenerate}
                    />
                  )
                })}

                {/* Streaming status timeline */}
                {loading && (
                  <StreamingTimeline
                    activeNode={activeNode}
                    visitedNodes={visitedNodes}
                    statusMessage={statusMessage}
                  />
                )}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
        </main>

        {/* Prominent Composer Area at Bottom */}
        <footer className="p-4 sm:p-5 bg-[var(--color-canvas)] border-t border-[var(--color-border-subtle)] shrink-0">
          <div className="max-w-[840px] mx-auto">
            <Composer
              input={input}
              onChange={setInput}
              onSend={handleSend}
              onStop={abortStream}
              loading={loading}
              maxLength={1500}
            />
          </div>
        </footer>
      </div>

      {/* Secondary Knowledge Base Inspector Dialog */}
      <Dialog
        open={kbDialogOpen}
        onClose={() => setKbDialogOpen(false)}
        title="Knowledge Base & Vector Index"
        description="Inspect Pinecone embeddings and query raw policy chunks"
        maxWidth="max-w-2xl"
      >
        <KBView
          onAskInChat={(query) => {
            setKbDialogOpen(false)
            handleSend(query)
          }}
        />
      </Dialog>

      {/* Secondary Settings Dialog */}
      <Dialog
        open={settingsDialogOpen}
        onClose={() => setSettingsDialogOpen(false)}
        title="System Architecture & Session Settings"
        description="Inspect LangGraph configuration and manage local thread checkpoints"
        maxWidth="max-w-xl"
      >
        <SettingsView
          backendOnline={backendOnline}
          onResetSession={() => {
            handleNewChat()
            setSettingsDialogOpen(false)
          }}
          onClearStorage={() => {
            clearStorage()
            handleNewChat()
            setSettingsDialogOpen(false)
          }}
        />
      </Dialog>
    </div>
  )
}

export function App() {
  return (
    <ErrorBoundary>
      <AppContent />
    </ErrorBoundary>
  )
}

export default App
