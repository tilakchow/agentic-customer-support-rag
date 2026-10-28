import React from "react"
import { CheckCircle2, Loader2 } from "lucide-react"

export interface StepItem {
  id: string
  label: string
  sublabel?: string
}

export interface StreamingTimelineProps {
  activeNode?: string
  visitedNodes?: string[]
  statusMessage?: string
  isComplete?: boolean
}

const TIMELINE_STAGES = [
  { id: "routing", label: "Classifying query", nodes: ["route_question"] },
  { id: "kb_retrieval", label: "Searching knowledge base", nodes: ["retrieve_kb"] },
  { id: "grading", label: "Grading policy evidence", nodes: ["grade_kb_evidence"] },
  { id: "web_fallback", label: "Web verification", nodes: ["search_web", "grade_web_evidence"] },
  { id: "writing", label: "Synthesizing answer", nodes: ["generate_from_kb", "generate_from_web", "direct_answer", "answer_insufficient"] },
]

export const StreamingTimeline: React.FC<StreamingTimelineProps> = ({
  activeNode,
  visitedNodes = [],
  statusMessage,
  isComplete = false,
}) => {
  const [collapsed, setCollapsed] = React.useState(false)

  // Determine stage states
  return (
    <div className="mb-4 rounded-[var(--radius-lg)] border border-[var(--color-border-subtle)] bg-[var(--color-surface)] p-3 text-[12px] text-[var(--color-ink-muted)]">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {!isComplete ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin text-[var(--color-brand)] shrink-0" />
          ) : (
            <CheckCircle2 className="w-3.5 h-3.5 text-[var(--color-success)] shrink-0" />
          )}
          <span className="font-medium text-[var(--color-ink)]">
            {isComplete ? "Reasoning complete" : statusMessage || "Analyzing..."}
          </span>
        </div>
        <button
          type="button"
          onClick={() => setCollapsed((prev) => !prev)}
          className="text-[11px] text-[var(--color-ink-faint)] hover:text-[var(--color-ink)] underline cursor-pointer select-none"
        >
          {collapsed ? "Show steps" : "Hide"}
        </button>
      </div>

      {!collapsed && (
        <div className="mt-2.5 pt-2 border-t border-[var(--color-border-subtle)] flex flex-wrap gap-2 text-[11px]">
          {TIMELINE_STAGES.map((stage) => {
            const isActive = activeNode ? stage.nodes.includes(activeNode) : false
            const isDone = stage.nodes.some((n) => visitedNodes.includes(n))

            return (
              <span
                key={stage.id}
                className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-[var(--radius-full)] border ${
                  isActive
                    ? "bg-[var(--color-brand-subtle)] text-[var(--color-brand)] border-[var(--color-brand)] font-medium"
                    : isDone
                    ? "bg-[var(--color-surface-hover)] text-[var(--color-ink-muted)] border-[var(--color-border-subtle)]"
                    : "opacity-40 border-transparent text-[var(--color-ink-faint)]"
                }`}
              >
                {isActive ? (
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-brand)] animate-ping" />
                ) : isDone ? (
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-success)]" />
                ) : (
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-ink-faint)]" />
                )}
                <span>{stage.label}</span>
              </span>
            )
          })}
        </div>
      )}
    </div>
  )
}
