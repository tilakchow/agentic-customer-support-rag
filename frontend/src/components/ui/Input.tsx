import React from "react"

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  error?: boolean
  icon?: React.ReactNode
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className = "", error, icon, disabled, ...props }, ref) => {
    return (
      <div className="relative flex items-center w-full">
        {icon && (
          <span className="absolute left-3 text-[var(--color-ink-muted)] pointer-events-none flex items-center">
            {icon}
          </span>
        )}
        <input
          ref={ref}
          disabled={disabled}
          className={`w-full bg-[var(--color-surface)] text-[var(--color-ink)] placeholder-[var(--color-ink-faint)] text-[13.5px] rounded-[var(--radius-sm)] border ${
            error
              ? "border-[var(--color-danger)] focus:border-[var(--color-danger)]"
              : "border-[var(--color-border)] focus:border-[var(--color-brand)]"
          } px-3 py-2 ${
            icon ? "pl-9" : ""
          } transition-colors focus:outline-none focus:ring-1 ${
            error ? "focus:ring-[var(--color-danger)]" : "focus:ring-[var(--color-brand)]"
          } disabled:opacity-50 disabled:bg-[var(--color-surface-hover)] ${className}`}
          {...props}
        />
      </div>
    )
  }
)

Input.displayName = "Input"

export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  error?: boolean
}

export const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className = "", error, disabled, ...props }, ref) => {
    return (
      <textarea
        ref={ref}
        disabled={disabled}
        className={`w-full bg-[var(--color-surface)] text-[var(--color-ink)] placeholder-[var(--color-ink-faint)] text-[14px] leading-relaxed rounded-[var(--radius-md)] border ${
          error
            ? "border-[var(--color-danger)] focus:border-[var(--color-danger)]"
            : "border-[var(--color-border)] focus:border-[var(--color-brand)]"
        } px-3.5 py-2.5 transition-colors focus:outline-none focus:ring-1 ${
          error ? "focus:ring-[var(--color-danger)]" : "focus:ring-[var(--color-brand)]"
        } disabled:opacity-50 resize-none ${className}`}
        {...props}
      />
    )
  }
)

Textarea.displayName = "Textarea"
