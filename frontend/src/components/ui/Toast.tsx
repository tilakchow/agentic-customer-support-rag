import React from "react"
import { AlertCircle, CheckCircle, Info, X } from "lucide-react"
import { IconButton } from "./IconButton"

export interface ToastProps {
  id: string
  title: string
  message?: string
  type?: "info" | "success" | "error"
  onClose: (id: string) => void
}

export const Toast: React.FC<ToastProps> = ({
  id,
  title,
  message,
  type = "info",
  onClose,
}) => {
  const icons = {
    info: <Info className="w-4 h-4 text-[var(--color-brand)] shrink-0" />,
    success: <CheckCircle className="w-4 h-4 text-[var(--color-success)] shrink-0" />,
    error: <AlertCircle className="w-4 h-4 text-[var(--color-danger)] shrink-0" />,
  }

  return (
    <div
      role="status"
      className="flex items-start gap-3 p-3.5 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-[var(--radius-md)] shadow-[var(--shadow-elevation)] text-[13px] text-[var(--color-ink)] min-w-[280px] max-w-sm animate-in slide-in-from-bottom-2 duration-150"
    >
      <div className="mt-0.5">{icons[type]}</div>
      <div className="flex-1 min-w-0">
        <p className="font-semibold leading-tight">{title}</p>
        {message && (
          <p className="text-[12px] text-[var(--color-ink-muted)] mt-0.5 leading-normal">
            {message}
          </p>
        )}
      </div>
      <IconButton
        variant="ghost"
        size="xs"
        aria-label="Dismiss alert"
        onClick={() => onClose(id)}
      >
        <X className="w-3.5 h-3.5" />
      </IconButton>
    </div>
  )
}
