import React, { useState, useEffect, useRef } from "react"
import { Paperclip, ChevronDown, Send, Loader2, AlertCircle } from "lucide-react"
import { Header } from "./components/Header"
import { Sidebar, NavTab } from "./components/Sidebar"
import { WelcomeSection } from "./components/WelcomeSection"
import { ChatMessage, MessageItem } from "./components/ChatMessage"
import { WorkflowPanel } from "./components/WorkflowPanel"
import { KBView } from "./components/views/KBView"
import { HistoryView, SavedThread } from "./components/views/HistoryView"
import { SettingsView } from "./components/views/SettingsView"
import { streamMessage, checkHealth } from "./services/api"

export function App() {
  const [messages, setMessages] = useState<MessageItem[]>([])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [statusMessage, setStatusMessage] = useState<string>("")
  const [activeNode, setActiveNode] = useState<string | undefined>(undefined)
  const [visitedNodes, setVisitedNodes] = useState<string[]>([])
  const [showWorkflow, setShowWorkflow] = useState(true)
  const [activeTab, setActiveTab] = useState<NavTab>("chat")
  const [searchFilter, setSearchFilter] = useState("")
  const [showSearch, setShowSearch] = useState(false)

  // Session Thread ID
  const [threadId, setThreadId] = useState<string>(() => {
    return localStorage.getItem("agentic_thread_id") || crypto.randomUUID()
  })
  const [backendOnline, setBackendOnline] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  // Saved Conversation Threads
  const [savedThreads, setSavedThreads] = useState<SavedThread[]>(() => {
    try {
      const stored = localStorage.getItem("agentic_saved_threads")
      return stored ? JSON.parse(stored) : []
    } catch {
      return []
    }
  })

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Sync threadId to localStorage
  useEffect(() => {
    localStorage.setItem("agentic_thread_id", threadId)
  }, [threadId])

  // Sync saved threads to localStorage
  useEffect(() => {
    try {
      localStorage.setItem("agentic_saved_threads", JSON.stringify(savedThreads))
    } catch (e) {
      console.error("Failed to save threads:", e)
    }
  }, [savedThreads])

  // Update current thread in savedThreads whenever messages update
  useEffect(() => {
    if (messages.length === 0) return

    const firstUserMsg = messages.find((m) => m.role === "user")?.content || "Customer Inquiry"
    const previewText = messages[messages.length - 1].content.slice(0, 90)
    const titleText = firstUserMsg.length > 45 ? `${firstUserMsg.slice(0, 45)}...` : firstUserMsg

    setSavedThreads((prev) => {
      const existingIdx = prev.findIndex((t) => t.threadId === threadId)
      const updatedItem: SavedThread = {
        threadId,
        title: titleText,
        preview: previewText,
        updatedAt: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
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
  }, [messages, threadId])

  // Periodic health check
  useEffect(() => {
    const verifyHealth = async () => {
      try {
        await checkHealth()
        setBackendOnline(true)
      } catch {
        setBackendOnline(false)
      }
    }

    verifyHealth()
    const interval = setInterval(verifyHealth, 10000)
    return () => clearInterval(interval)
  }, [])

  // Auto scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    if (activeTab === "chat") {
      scrollToBottom()
    }
  }, [messages, loading, statusMessage, activeTab])

  // Handle New Chat
  const handleNewChat = () => {
    const newId = crypto.randomUUID()
    setThreadId(newId)
    setMessages([])
    setErrorMsg(null)
    setInput("")
    setStatusMessage("")
    setActiveNode(undefined)
    setVisitedNodes([])
    setActiveTab("chat")
    setShowSearch(false)
    setSearchFilter("")
  }

  // Resume past thread from History view
  const handleResumeThread = (thread: SavedThread) => {
    setThreadId(thread.threadId)
    setMessages(thread.messages)
    setActiveTab("chat")
    setActiveNode(undefined)
    setVisitedNodes([])
  }

  // Delete thread
  const handleDeleteThread = (delId: string) => {
    setSavedThreads((prev) => prev.filter((t) => t.threadId !== delId))
    if (delId === threadId) {
      handleNewChat()
    }
  }

  // Clear all threads
  const handleClearAllThreads = () => {
    setSavedThreads([])
    handleNewChat()
  }

  // Reset Storage from Settings
  const handleClearStorage = () => {
    localStorage.removeItem("agentic_saved_threads")
    localStorage.removeItem("agentic_thread_id")
    setSavedThreads([])
    handleNewChat()
  }

  // Send message with streaming
  const handleSend = async (textToSend?: string) => {
    const query = (textToSend || input).trim()
    if (!query || loading) return

    setActiveTab("chat")
    setErrorMsg(null)
    setActiveNode("route_question")
    setVisitedNodes(["route_question"])

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
    setLoading(true)
    setStatusMessage("Routing question...")

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
    }

    try {
      await streamMessage(
        {
          message: query,
          thread_id: threadId,
        },
        {
          onStatus: (status, node) => {
            setStatusMessage(status)
            if (node) {
              setActiveNode(node)
              setVisitedNodes((prev) => Array.from(new Set([...prev, node])))
            }
          },
          onToken: (token) => {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId ? { ...msg, content: msg.content + token } : msg
              )
            )
          },
          onDone: (data) => {
            setMessages((prev) =>
              prev.map((msg) => {
                if (msg.id === assistantId) {
                  return {
                    ...msg,
                    content: msg.content || data.full_content || "",
                    source: data.source_used,
                  }
                }
                return msg
              })
            )
            if (data.thread_id && data.thread_id !== threadId) {
              setThreadId(data.thread_id)
            }
            setActiveNode(undefined)
            setStatusMessage("")
            setLoading(false)
          },
          onError: (err) => {
            setErrorMsg(err)
            setActiveNode(undefined)
            setStatusMessage("")
            setLoading(false)
            setMessages((prev) => prev.filter((msg) => msg.id !== assistantId || msg.content.length > 0))
          },
        }
      )
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to reach the agent. Please verify that FastAPI is running."
      setErrorMsg(msg)
      setActiveNode(undefined)
      setMessages((prev) => prev.filter((msg) => msg.id !== assistantId || msg.content.length > 0))
    } finally {
      setLoading(false)
      setActiveNode(undefined)
      setStatusMessage("")
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
    e.target.style.height = "auto"
    e.target.style.height = `${Math.min(e.target.scrollHeight, 140)}px`
  }

  // Filter messages if search is active
  const displayedMessages = searchFilter
    ? messages.filter((m) => m.content.toLowerCase().includes(searchFilter.toLowerCase()))
    : messages

  return (
    <div className="flex flex-col h-screen w-screen bg-[#f8fafc] text-slate-900 font-sans overflow-hidden">
      {/* Top Header */}
      <Header
        threadId={threadId}
        backendOnline={backendOnline}
        showWorkflow={showWorkflow}
        onToggleWorkflow={() => setShowWorkflow((prev) => !prev)}
        onNewChat={handleNewChat}
        onSearchClick={() => {
          if (activeTab !== "chat") setActiveTab("chat")
          setShowSearch((prev) => !prev)
        }}
      />

      {/* Main Workspace with Slim Sidebar */}
      <div className="flex-1 flex overflow-hidden">
        {/* Slim Left Sidebar (Fully Functional Tabs!) */}
        <Sidebar
          activeTab={activeTab}
          onSelectTab={(tab) => setActiveTab(tab)}
          onHelpClick={() => {
            handleSend("What topics and customer support procedures are in your knowledge base?")
          }}
        />

        {/* Central Dynamic Content Area */}
        <div className="flex-1 flex overflow-hidden">
          {/* TAB 1: Chat View */}
          {activeTab === "chat" && (
            <div className="flex-1 flex flex-col justify-between overflow-hidden relative">
              {/* Optional Search Bar in Chat */}
              {showSearch && (
                <div className="px-6 py-2.5 bg-blue-50/70 border-b border-blue-100 flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2 flex-1 max-w-md">
                    <span className="text-slate-500 font-medium">Find in conversation:</span>
                    <input
                      type="text"
                      value={searchFilter}
                      onChange={(e) => setSearchFilter(e.target.value)}
                      placeholder="Type keywords..."
                      className="px-2.5 py-1 rounded-lg bg-white border border-slate-300 text-xs focus:outline-none focus:border-blue-500 flex-1"
                    />
                  </div>
                  <button
                    onClick={() => {
                      setSearchFilter("")
                      setShowSearch(false)
                    }}
                    className="text-slate-400 hover:text-slate-700 text-xs font-medium cursor-pointer"
                  >
                    Close
                  </button>
                </div>
              )}

              {/* Scrollable Message Viewport */}
              <main className="flex-1 overflow-y-auto px-4 md:px-8 py-5">
                <div className="max-w-3xl mx-auto flex flex-col min-h-full">
                  {/* Error Banner */}
                  {errorMsg && (
                    <div className="mb-4 flex items-center gap-2 p-3 rounded-xl border border-red-200 bg-red-50 text-red-700 text-xs">
                      <AlertCircle className="w-4 h-4 shrink-0 text-red-600" />
                      <span className="flex-1">{errorMsg}</span>
                      <button
                        onClick={() => setErrorMsg(null)}
                        className="text-red-500 hover:text-red-800 text-[11px] underline cursor-pointer"
                      >
                        Dismiss
                      </button>
                    </div>
                  )}

                  {/* Welcome section: visible only before conversation starts */}
                  {messages.length === 0 ? (
                    <div className="my-auto py-4">
                      <WelcomeSection onSelectQuery={(query) => handleSend(query)} />
                    </div>
                  ) : (
                    <div className="flex-1 flex flex-col pb-6">
                      {displayedMessages.map((msg) => {
                        if (msg.role === "assistant" && !msg.content && loading) {
                          return null
                        }
                        return <ChatMessage key={msg.id} message={msg} />
                      })}

                      {/* Live streaming status card */}
                      {loading && statusMessage && (
                        <div className="flex items-center gap-3 p-3 mb-4 rounded-xl border border-slate-200/90 bg-white shadow-2xs max-w-sm text-slate-700 text-xs ml-11">
                          <Loader2 className="w-3.5 h-3.5 animate-spin text-blue-600 shrink-0" />
                          <span className="font-medium">{statusMessage}</span>
                        </div>
                      )}
                      <div ref={messagesEndRef} />
                    </div>
                  )}
                </div>
              </main>

              {/* Bottom Input Area matching reference design */}
              <div className="p-4 bg-gradient-to-t from-[#f8fafc] via-[#f8fafc]/90 to-transparent">
                <div className="max-w-3xl mx-auto">
                  <div className="flex items-center gap-2 rounded-2xl border border-slate-300/90 bg-white px-3.5 py-2 shadow-xs focus-within:border-blue-500 focus-within:ring-1 focus-within:ring-blue-500 transition-all">
                    {/* Attachment button */}
                    <button
                      type="button"
                      onClick={() => {
                        handleSend("How do I upload or reference custom order receipts?")
                      }}
                      className="p-1 text-slate-400 hover:text-slate-700 rounded transition-colors cursor-pointer shrink-0"
                      title="Attach file / receipts"
                    >
                      <Paperclip className="w-4 h-4" />
                    </button>

                    {/* Textarea input */}
                    <textarea
                      ref={textareaRef}
                      rows={1}
                      value={input}
                      onChange={handleTextareaChange}
                      onKeyDown={handleKeyDown}
                      placeholder="Ask a support question or search the knowledge base..."
                      className="flex-1 resize-none bg-transparent px-2 py-1 text-[13.5px] text-slate-900 placeholder-slate-400 focus:outline-none max-h-32 leading-relaxed"
                    />

                    {/* Source Mode Pill (Web + KB) */}
                    <div
                      className="hidden sm:flex items-center gap-1 px-2 py-1 rounded-lg bg-slate-100/90 border border-slate-200/80 text-[11px] font-medium text-slate-600 shrink-0 cursor-pointer hover:bg-slate-200/70 transition-colors"
                      title="Autonomous Routing: Searches Pinecone Knowledge Base with Tavily Web fallback"
                    >
                      <span>Web + KB</span>
                      <ChevronDown className="w-3 h-3 text-slate-400" />
                    </div>

                    {/* Send Button */}
                    <button
                      onClick={() => handleSend()}
                      disabled={!input.trim() || loading}
                      className="w-8 h-8 rounded-full bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white flex items-center justify-center transition-all disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer shrink-0 shadow-2xs"
                      title="Send message"
                    >
                      <Send className="w-3.5 h-3.5 fill-current ml-0.5" />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: KB (Knowledge Base Inspector View) */}
          {activeTab === "kb" && (
            <KBView
              onAskInChat={(query) => {
                handleSend(query)
              }}
            />
          )}

          {/* TAB 3: History View */}
          {activeTab === "history" && (
            <HistoryView
              currentThreadId={threadId}
              threads={savedThreads}
              onResumeThread={handleResumeThread}
              onDeleteThread={handleDeleteThread}
              onClearAll={handleClearAllThreads}
              onNewChat={handleNewChat}
            />
          )}

          {/* TAB 4: Settings View */}
          {activeTab === "settings" && (
            <SettingsView
              backendOnline={backendOnline}
              onResetSession={handleNewChat}
              onClearStorage={handleClearStorage}
            />
          )}

          {/* Right-Side LangGraph Workflow Panel (Visible in Chat view) */}
          {activeTab === "chat" && showWorkflow && (
            <aside className="border-l border-slate-200/90 bg-white/50 p-4 hidden md:flex items-start overflow-y-auto">
              <WorkflowPanel
                activeNode={activeNode}
                visitedNodes={visitedNodes}
                onClose={() => setShowWorkflow(false)}
              />
            </aside>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
