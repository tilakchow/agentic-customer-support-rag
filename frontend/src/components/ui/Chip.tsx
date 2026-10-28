import React from "react"

export interface ChipProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  active?: boolean
  icon?: React.ReactNode
}

export const Chip: React.FC<ChipProps> = ({
  children,
  active = false,
  icon,
  className = "",
  disabled,
  ...props
}) => {
  return (
    <button
      disabled={disabled}
      className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-[var(--radius-full)] text-[12px] font-medium transition-colors select-none cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed ${
        active
          ? "bg-[var(--color-brand)] text-[var(--color-brand-text)] shadow-xs"
          : "bg-[var(--color-surface)] text-[var(--color-ink)] border border-[var(--color-border)] hover:bg-[var(--color-surface-hover)]"
      } ${className}`}
      {...props}
    >
      {icon && <span className="shrink-0 flex items-center">{icon}</span>}
      <span>{children}</span>
    </button>
  )
}
