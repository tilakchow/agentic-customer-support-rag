import React from "react"

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost" | "danger"
  size?: "sm" | "md" | "lg"
  icon?: React.ReactNode
  iconRight?: React.ReactNode
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      children,
      variant = "secondary",
      size = "md",
      icon,
      iconRight,
      className = "",
      disabled,
      ...props
    },
    ref
  ) => {
    const base =
      "inline-flex items-center justify-center font-medium transition-colors select-none cursor-pointer disabled:opacity-45 disabled:cursor-not-allowed focus-visible:outline-2 focus-visible:outline-offset-2"

    const sizeClasses = {
      sm: "text-[12px] h-7 px-2.5 gap-1.5 rounded-[var(--radius-sm)]",
      md: "text-[13.5px] h-9 px-3.5 gap-2 rounded-[var(--radius-sm)]",
      lg: "text-[15px] h-11 px-5 gap-2.5 rounded-[var(--radius-md)]",
    }

    const variantClasses = {
      primary:
        "bg-[var(--color-ink)] text-[var(--color-canvas)] hover:opacity-90 active:scale-[0.98] border border-transparent shadow-[var(--shadow-card)]",
      secondary:
        "bg-[var(--color-surface)] text-[var(--color-ink)] border border-[var(--color-border)] hover:bg-[var(--color-surface-hover)] active:bg-[var(--color-surface-active)] shadow-[var(--shadow-card)]",
      ghost:
        "bg-transparent text-[var(--color-ink-muted)] hover:text-[var(--color-ink)] hover:bg-[var(--color-surface-hover)] border border-transparent",
      danger:
        "bg-[var(--color-danger-subtle)] text-[var(--color-danger)] border border-[var(--color-danger-border)] hover:opacity-90",
    }

    return (
      <button
        ref={ref}
        disabled={disabled}
        className={`${base} ${sizeClasses[size]} ${variantClasses[variant]} ${className}`}
        {...props}
      >
        {icon && <span className="shrink-0 flex items-center">{icon}</span>}
        {children && <span>{children}</span>}
        {iconRight && <span className="shrink-0 flex items-center">{iconRight}</span>}
      </button>
    )
  }
)

Button.displayName = "Button"
