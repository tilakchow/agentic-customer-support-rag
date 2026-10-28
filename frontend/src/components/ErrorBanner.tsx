import React from "react"
import { AlertCircle, RefreshCw, WifiOff } from "lucide-react"

export interface ErrorBannerProps {
  message: string
  onRetry?: () => void
  onDismiss?: () => void
}

export const ErrorBanner: React.FC<ErrorBannerProps> = ({
  message,
  onRetry,
  onDismiss,
}) => {
  const isOffline = message.toLowerCase().includes("offline") || message.toLowerCase().includes("failed to fetch")
  const isRateLimit = message.includes("429") || message.toLowerCase().includes("rate limit")

  let title = "Inquiry Error"
  let description = message

  if (isOffline) {
    title = "Backend Unreachable"
    description = "Could not connect to the assistant server. Ensure FastAPI is active and check your network."
  } else if (isRateLimit) {
    title = "Rate Limit Reached"
    description = "Too many requests submitted in a short period. Please wait a moment before sending again."
  }

  return (
    <div
      role="alert"
      className="mb-5 p-3.5 rounded-[var(--radius-lg)] border border-[var(--color-danger-border)] bg-[var(--color-danger-subtle)] text-[var(--color-danger)] text-[13px] flex items-start gap-3 shadow-xs select-none"
    >
      {isOffline ? (
        <WifiOff className="w-4 h-4 shrink-0 mt-0.5 text-[var(--color-danger)]" />
      ) : (
        <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-[var(--color-danger)]" />
      )}
      <div className="flex-1 min-w-0">
        <p className="font-semibold">{title}</p>
        <p className="text-[12px] opacity-90 mt-0.5 leading-relaxed">{description}</p>
      </div>

      <div className="flex items-center gap-2 shrink-0">
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="inline-flex items-center gap-1 px-2.5 py-1 rounded-[var(--radius-xs)] bg-[var(--color-surface)] border border-[var(--color-danger-border)] text-[var(--color-danger)] text-[11px] font-medium hover:opacity-90 transition-opacity cursor-pointer"
          >
            <RefreshCw className="w-3 h-3" />
            <span>Retry</span>
          </button>
        )}
        {onDismiss && (
          <button
            type="button"
            onClick={onDismiss}
            className="text-[11px] opacity-75 hover:opacity-100 underline cursor-pointer"
          >
            Dismiss
          </button>
        )}
      </div>
    </div>
  )
}
