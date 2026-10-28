import React from "react"

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "elevated" | "interactive"
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ children, variant = "default", className = "", ...props }, ref) => {
    const base = "bg-[var(--color-surface)] border border-[var(--color-border)] rounded-[var(--radius-xl)] transition-all"
    
    const variantStyles = {
      default: "shadow-[var(--shadow-card)]",
      elevated: "shadow-[var(--shadow-elevation)]",
      interactive:
        "shadow-[var(--shadow-card)] hover:border-[var(--color-brand)] hover:shadow-[var(--shadow-elevation)] cursor-pointer",
    }

    return (
      <div
        ref={ref}
        className={`${base} ${variantStyles[variant]} ${className}`}
        {...props}
      >
        {children}
      </div>
    )
  }
)

Card.displayName = "Card"
