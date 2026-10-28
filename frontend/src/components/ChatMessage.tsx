import React, { useState, useMemo } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { Copy, Check, ThumbsUp, ThumbsDown, RotateCcw } from "lucide-react"
import { SourcesSection } from "./SourcesSection"
import { CitationSource } from "../services/api"

export interface MessageItem {
  id: string
  role: "user" | "assistant"
  content: string
  source?: string
  sources?: CitationSource[]
  timestamp: string
}

interface ChatMessageProps {
  message: MessageItem
  onRegenerate?: () => void
  isLastAssistant?: boolean
}

export const ChatMessage: React.FC<ChatMessageProps> = ({
  message,
  onRegenerate,
  isLastAssistant = false,
}) => {
  const [copied, setCopied] = useState(false)
  const [feedback, setFeedback] = useState<"up" | "down" | null>(() => {
    try {
      const stored = localStorage.getItem(`feedback_${message.id}`)
      return (stored as "up" | "down") || null
    } catch {
      return null
    }
  })

  const isUser = message.role === "user"

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleFeedback = (type: "up" | "down") => {
    const next = feedback === type ? null : type
    setFeedback(next)
    try {
      if (next) {
        localStorage.setItem(`feedback_${message.id}`, next)
      } else {
        localStorage.removeItem(`feedback_${message.id}`)
      }
    } catch (e) {
      console.warn("Could not save feedback to localStorage:", e)
    }
  }

  const formattedContent = useMemo(() => {
    if (!message.content) return ""
    return message.content.replace(/<br\s*\/?>/gi, "\n")
  }, [message.content])

  if (isUser) {
    return (
      <div className="flex justify-end mb-6">
        <div className="max-w-[85%] sm:max-w-[75%] rounded-[var(--radius-lg)] bg-[var(--color-surface-hover)] border border-[var(--color-border)] px-4 py-3 text-[14px] leading-relaxed text-[var(--color-ink)] shadow-[var(--shadow-card)]">
          <p className="whitespace-pre-wrap">{message.content}</p>
          <div className="text-right mt-1">
            <span className="text-[10px] text-[var(--color-ink-faint)] font-mono">
              {message.timestamp}
            </span>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="mb-8 group">
      {/* Assistant Answer Box (Answer-first layout, full centered column reading experience) */}
      <div className="rounded-[var(--radius-xl)] bg-[var(--color-surface)] border border-[var(--color-border)] p-5 sm:p-6 shadow-[var(--shadow-card)] transition-all">
        {/* Markdown content with clean Perplexity typography */}
        <div className="prose prose-neutral dark:prose-invert max-w-none text-[14.5px] leading-relaxed text-[var(--color-ink)]">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              h1: ({ children }) => (
                <h1 className="text-[18px] font-semibold text-[var(--color-ink)] mt-4 mb-2 first:mt-0 tracking-tight">
                  {children}
                </h1>
              ),
              h2: ({ children }) => (
                <h2 className="text-[16px] font-semibold text-[var(--color-ink)] mt-3.5 mb-1.5 first:mt-0 tracking-tight">
                  {children}
                </h2>
              ),
              h3: ({ children }) => (
                <h3 className="text-[14.5px] font-semibold text-[var(--color-ink)] mt-3 mb-1 first:mt-0">
                  {children}
                </h3>
              ),
              p: ({ children }) => (
                <p className="mb-3 last:mb-0 leading-relaxed text-[var(--color-ink)]">
                  {children}
                </p>
              ),
              strong: ({ children }) => (
                <strong className="font-semibold text-[var(--color-ink)]">{children}</strong>
              ),
              em: ({ children }) => (
                <em className="italic text-[var(--color-ink-muted)]">{children}</em>
              ),
              ul: ({ children }) => (
                <ul className="list-disc pl-5 my-2.5 space-y-1 text-[var(--color-ink)]">
                  {children}
                </ul>
              ),
              ol: ({ children }) => (
                <ol className="list-decimal pl-5 my-2.5 space-y-1 text-[var(--color-ink)]">
                  {children}
                </ol>
              ),
              li: ({ children }) => (
                <li className="leading-relaxed pl-1">{children}</li>
              ),
              blockquote: ({ children }) => (
                <blockquote className="border-l-2 border-[var(--color-brand)] pl-3.5 my-3 text-[var(--color-ink-muted)] italic">
                  {children}
                </blockquote>
              ),
              table: ({ children }) => (
                <div className="my-4 w-full overflow-x-auto rounded-[var(--radius-md)] border border-[var(--color-border)]">
                  <table className="w-full min-w-[400px] text-left text-[13px] border-collapse">
                    {children}
                  </table>
                </div>
              ),
              thead: ({ children }) => (
                <thead className="bg-[var(--color-surface-hover)] border-b border-[var(--color-border)] font-semibold text-[var(--color-ink)]">
                  {children}
                </thead>
              ),
              tbody: ({ children }) => (
                <tbody className="divide-y divide-[var(--color-border-subtle)] bg-[var(--color-surface)]">
                  {children}
                </tbody>
              ),
              tr: ({ children }) => (
                <tr className="hover:bg-[var(--color-surface-hover)] transition-colors">
                  {children}
                </tr>
              ),
              th: ({ children }) => (
                <th className="px-3.5 py-2 font-semibold text-[var(--color-ink)] border-r border-[var(--color-border-subtle)] last:border-r-0">
                  {children}
                </th>
              ),
              td: ({ children }) => (
                <td className="px-3.5 py-2 text-[var(--color-ink-muted)] align-top border-r border-[var(--color-border-subtle)] last:border-r-0">
                  {children}
                </td>
              ),
              code: ({ className, children, ...props }) => {
                const match = /language-(\w+)/.exec(className || "")
                const isMultiline = String(children).includes("\n")

                if (!match && !isMultiline) {
                  return (
                    <code
                      className="px-1.5 py-0.5 rounded-[var(--radius-xs)] bg-[var(--color-surface-hover)] text-[var(--color-ink)] font-mono text-[12.5px] border border-[var(--color-border)]"
                      {...props}
                    >
                      {children}
                    </code>
                  )
                }

                return (
                  <div className="my-3 rounded-[var(--radius-md)] overflow-hidden border border-[var(--color-border)] bg-[#1e1d1a] text-slate-100">
                    {match && (
                      <div className="px-3 py-1 bg-[#282723] border-b border-[#363530] text-[11px] text-slate-400 font-mono">
                        {match[1]}
                      </div>
                    )}
                    <pre className="p-3.5 overflow-x-auto text-[12.5px] font-mono leading-relaxed text-slate-200">
                      <code className={className} {...props}>
                        {children}
                      </code>
                    </pre>
                  </div>
                )
              },
            }}
          >
            {formattedContent}
          </ReactMarkdown>
        </div>

        {/* Sources section under each answer */}
        <SourcesSection
          sourceUsed={message.source}
          sources={message.sources}
        />
      </div>

      {/* Answer Action Bar: Copy, Thumbs, Regenerate, Timestamp */}
      <div className="flex items-center justify-between px-2 mt-2 text-[12px] text-[var(--color-ink-faint)]">
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={handleCopy}
            className="inline-flex items-center gap-1 px-2 py-1 rounded-[var(--radius-sm)] hover:text-[var(--color-ink)] hover:bg-[var(--color-surface-hover)] transition-colors cursor-pointer"
            aria-label="Copy answer text"
            title="Copy answer"
          >
            {copied ? (
              <Check className="w-3.5 h-3.5 text-[var(--color-success)]" />
            ) : (
              <Copy className="w-3.5 h-3.5" />
            )}
            <span className="text-[11px]">{copied ? "Copied" : "Copy"}</span>
          </button>

          <button
            type="button"
            onClick={() => handleFeedback("up")}
            className={`p-1.5 rounded-[var(--radius-sm)] hover:bg-[var(--color-surface-hover)] transition-colors cursor-pointer ${
              feedback === "up" ? "text-[var(--color-brand)]" : "hover:text-[var(--color-ink)]"
            }`}
            aria-label="Helpful answer"
            title="Helpful"
          >
            <ThumbsUp className="w-3.5 h-3.5" />
          </button>

          <button
            type="button"
            onClick={() => handleFeedback("down")}
            className={`p-1.5 rounded-[var(--radius-sm)] hover:bg-[var(--color-surface-hover)] transition-colors cursor-pointer ${
              feedback === "down" ? "text-[var(--color-danger)]" : "hover:text-[var(--color-ink)]"
            }`}
            aria-label="Not helpful answer"
            title="Not helpful"
          >
            <ThumbsDown className="w-3.5 h-3.5" />
          </button>

          {isLastAssistant && onRegenerate && (
            <button
              type="button"
              onClick={onRegenerate}
              className="inline-flex items-center gap-1 px-2 py-1 rounded-[var(--radius-sm)] hover:text-[var(--color-ink)] hover:bg-[var(--color-surface-hover)] transition-colors cursor-pointer ml-1"
              aria-label="Regenerate answer"
              title="Regenerate response"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span className="text-[11px]">Regenerate</span>
            </button>
          )}
        </div>

        <span className="font-mono text-[10.5px] text-[var(--color-ink-faint)]">
          {message.timestamp}
        </span>
      </div>
    </div>
  )
}
