import React from "react"

export interface IconButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  "aria-label": string
  variant?: "primary" | "secondary" | "ghost" | "brand"
  size?: "xs" | "sm" | "md" | "lg"
}

export const IconButton = React.forwardRef<HTMLButtonElement, IconButtonProps>(
  (
    {
      children,
      variant = "ghost",
      size = "md",
      className = "",
      disabled,
      ...props
    },
    ref
  ) => {
    const base =
      "inline-flex items-center justify-center shrink-0 transition-colors select-none cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed focus-visible:outline-2 focus-visible:outline-offset-2"

    const sizeClasses = {
      xs: "w-6 h-6 rounded-[var(--radius-xs)] text-[12px]",
      sm: "w-7 h-7 rounded-[var(--radius-sm)] text-[13px]",
      md: "w-9 h-9 rounded-[var(--radius-sm)] text-[14px]",
      lg: "w-11 h-11 rounded-[var(--radius-md)] text-[16px]",
    }

    const variantClasses = {
      primary:
        "bg-[var(--color-ink)] text-[var(--color-canvas)] hover:opacity-90 active:scale-[0.97]",
      secondary:
        "bg-[var(--color-surface)] text-[var(--color-ink)] border border-[var(--color-border)] hover:bg-[var(--color-surface-hover)] active:bg-[var(--color-surface-active)] shadow-[var(--shadow-card)]",
      ghost:
        "bg-transparent text-[var(--color-ink-muted)] hover:text-[var(--color-ink)] hover:bg-[var(--color-surface-hover)]",
      brand:
        "bg-[var(--color-brand)] text-[var(--color-brand-text)] hover:bg-[var(--color-brand-hover)] active:scale-[0.97]",
    }

    return (
      <button
        ref={ref}
        disabled={disabled}
        className={`${base} ${sizeClasses[size]} ${variantClasses[variant]} ${className}`}
        {...props}
      >
        {children}
      </button>
    )
  }
)

IconButton.displayName = "IconButton"
