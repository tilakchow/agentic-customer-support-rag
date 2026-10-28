import React from "react"
import { Database } from "lucide-react"

export interface ComposerProps {
  input: string
  onChange: (val: string) => void
  onSend: (text?: string) => void
  onStop?: () => void
  loading: boolean
  maxLength?: number
  placeholder?: string
}

export const Composer: React.FC<ComposerProps> = ({
  input,
  onChange,
  onSend,
  onStop,
  loading,
  maxLength = 1500,
  placeholder = "Ask a question about refund policies, order changes, account access...",
}) => {
  const textareaRef = React.useRef<HTMLTextAreaElement>(null)

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      if (input.trim() && !loading && input.length <= maxLength) {
        onSend()
      }
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e.target.value)
    e.target.style.height = "auto"
    e.target.style.height = `${Math.min(e.target.scrollHeight, 180)}px`
  }

  const charCount = input.length
  const isOverLimit = charCount > maxLength
  const isNearLimit = charCount > maxLength - 150

  return (
    <div className="w-full relative">
      <div
        className={`w-full rounded-[var(--radius-xl)] bg-[var(--color-surface)] border transition-all ${
          isOverLimit
            ? "border-[var(--color-danger)] ring-1 ring-[var(--color-danger)]"
            : "border-[var(--color-border)] focus-within:border-[var(--color-brand)] focus-within:ring-1 focus-within:ring-[var(--color-brand)] shadow-[var(--shadow-card)] hover:border-[var(--color-ink-faint)]"
        } p-3 sm:p-3.5 flex flex-col gap-2`}
      >
        <textarea
          ref={textareaRef}
          rows={1}
          value={input}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          aria-label="Ask a customer support question"
          className="w-full bg-transparent text-[14px] leading-relaxed text-[var(--color-ink)] placeholder-[var(--color-ink-faint)] focus:outline-none resize-none max-h-44"
        />

        {/* Footer controls */}
        <div className="flex items-center justify-between pt-1 select-none text-[12px] text-[var(--color-ink-muted)]">
          {/* Autonomous indicator (purely informational chip, not pretending to be a fake upload or select) */}
          <div className="flex items-center gap-1.5 text-[11px] text-[var(--color-ink-faint)]">
            <span className="flex items-center gap-1 px-2 py-0.5 rounded-[var(--radius-full)] bg-[var(--color-surface-hover)] border border-[var(--color-border-subtle)] font-medium text-[var(--color-ink-muted)]">
              <Database className="w-3 h-3 text-[var(--color-brand)]" />
              <span>Pinecone KB + Tavily</span>
            </span>
          </div>

          <div className="flex items-center gap-3">
            {/* Character count warning */}
            {charCount > 0 && (
              <span
                className={`font-mono text-[11px] ${
                  isOverLimit
                    ? "text-[var(--color-danger)] font-semibold"
                    : isNearLimit
                    ? "text-amber-600 dark:text-amber-400"
                    : "text-[var(--color-ink-faint)]"
                }`}
              >
                {charCount}/{maxLength}
              </span>
            )}

            {/* Action button: Send or Stop */}
            {loading ? (
              <button
                type="button"
                onClick={onStop}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-[var(--radius-sm)] bg-[var(--color-ink)] text-[var(--color-canvas)] text-[12px] font-medium hover:opacity-90 active:scale-95 transition-all cursor-pointer"
                aria-label="Stop generating answer"
              >
                <span className="w-2 h-2 rounded-xs bg-[var(--color-canvas)]" />
                <span>Stop</span>
              </button>
            ) : (
              <button
                type="button"
                onClick={() => onSend()}
                disabled={!input.trim() || isOverLimit}
                className="w-8 h-8 rounded-[var(--radius-full)] bg-[var(--color-ink)] text-[var(--color-canvas)] flex items-center justify-center hover:opacity-90 active:scale-95 transition-all disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer shadow-xs"
                aria-label="Send message"
              >
                <svg
                  className="w-3.5 h-3.5"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <line x1="12" y1="19" x2="12" y2="5" />
                  <polyline points="5 12 12 5 19 12" />
                </svg>
              </button>
            )}
          </div>
        </div>
      </div>

      {isOverLimit && (
        <p className="mt-1.5 text-[11.5px] text-[var(--color-danger)] px-1">
          Inquiry exceeds maximum supported limit of {maxLength} characters. Please condense your question.
        </p>
      )}
    </div>
  )
}
