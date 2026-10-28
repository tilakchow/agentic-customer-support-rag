import React from "react"
import { Database, Globe, HelpCircle, FileText } from "lucide-react"
import { CitationSource } from "../services/api"

export interface SourcesSectionProps {
  sourceUsed?: string
  sources?: CitationSource[]
}

export const SourcesSection: React.FC<SourcesSectionProps> = ({
  sourceUsed,
  sources,
}) => {
  const [expandedDoc, setExpandedDoc] = React.useState<string | null>(null)

  // Badge mapping for source_used
  const getBadgeDetails = (src?: string) => {
    switch (src) {
      case "private_kb":
      case "kb":
        return {
          label: "Knowledge Base",
          icon: <Database className="w-3.5 h-3.5 text-[var(--color-brand)]" />,
          desc: "Grounding verified against corporate customer support procedure records.",
        }
      case "web_search":
      case "web":
        return {
          label: "Live Web Verification",
          icon: <Globe className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />,
          desc: "Retrieved via Tavily search when internal evidence was insufficient.",
        }
      case "insufficient_evidence":
        return {
          label: "Insufficient Evidence",
          icon: <HelpCircle className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400" />,
          desc: "Inquiry could not be fully substantiated by verified documentation.",
        }
      default:
        return {
          label: "Direct Guidance",
          icon: <FileText className="w-3.5 h-3.5 text-[var(--color-ink-muted)]" />,
          desc: "General conversational support response.",
        }
    }
  }

  const badge = getBadgeDetails(sourceUsed)

  if (!sourceUsed && (!sources || sources.length === 0)) {
    return null
  }

  return (
    <div className="mt-5 pt-4 border-t border-[var(--color-border-subtle)] text-[12px] select-none">
      <div className="flex items-center gap-2 mb-2.5">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--color-ink-faint)]">
          Sources & Attribution
        </span>
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-[var(--radius-full)] bg-[var(--color-surface-hover)] border border-[var(--color-border-subtle)] font-medium text-[var(--color-ink)]">
          {badge.icon}
          <span>{badge.label}</span>
        </span>
      </div>

      {/* Render Citation cards if sources array exists (from SSE done event) */}
      {sources && sources.length > 0 ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-2">
          {sources.map((item, idx) => {
            const isSelected = expandedDoc === item.doc_id
            return (
              <div
                key={item.doc_id || idx}
                onClick={() => setExpandedDoc(isSelected ? null : item.doc_id)}
                className={`p-2.5 rounded-[var(--radius-lg)] border bg-[var(--color-surface)] transition-all cursor-pointer ${
                  isSelected
                    ? "border-[var(--color-brand)] ring-1 ring-[var(--color-brand)]"
                    : "border-[var(--color-border)] hover:border-[var(--color-ink-faint)] hover:bg-[var(--color-surface-hover)]"
                }`}
              >
                <div className="flex items-center justify-between text-[11px] text-[var(--color-ink-muted)] mb-1">
                  <span className="font-medium text-[var(--color-ink)]">
                    {item.intent ? item.intent.replace(/_/g, " ") : `Document #${idx + 1}`}
                  </span>
                  {item.score !== undefined && (
                    <span className="font-mono text-[10px] text-[var(--color-ink-faint)]">
                      {(item.score * 100).toFixed(0)}% match
                    </span>
                  )}
                </div>
                <p className="text-[11.5px] leading-relaxed text-[var(--color-ink-muted)] line-clamp-2">
                  {item.content}
                </p>
                {isSelected && (
                  <p className="mt-2 pt-2 border-t border-[var(--color-border-subtle)] text-[11px] leading-relaxed text-[var(--color-ink)] whitespace-pre-wrap">
                    {item.content}
                  </p>
                )}
              </div>
            )
          })}
        </div>
      ) : (
        <p className="text-[11.5px] text-[var(--color-ink-muted)] leading-relaxed">
          {badge.desc}
        </p>
      )}
    </div>
  )
}
