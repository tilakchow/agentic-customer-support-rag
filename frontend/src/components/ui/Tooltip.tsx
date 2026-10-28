import React, { useState } from "react"

export interface TooltipProps {
  content: string
  children: React.ReactElement
  side?: "top" | "bottom" | "left" | "right"
}

export const Tooltip: React.FC<TooltipProps> = ({
  content,
  children,
  side = "top",
}) => {
  const [visible, setVisible] = useState(false)

  const positions = {
    top: "bottom-full left-1/2 -translate-x-1/2 mb-1.5",
    bottom: "top-full left-1/2 -translate-x-1/2 mt-1.5",
    left: "right-full top-1/2 -translate-y-1/2 mr-1.5",
    right: "left-full top-1/2 -translate-y-1/2 ml-1.5",
  }

  return (
    <div
      className="relative inline-flex items-center"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
      onFocus={() => setVisible(true)}
      onBlur={() => setVisible(false)}
    >
      {children}
      {visible && (
        <div
          role="tooltip"
          className={`absolute z-50 whitespace-nowrap px-2 py-1 bg-[var(--color-ink)] text-[var(--color-canvas)] text-[11px] font-medium rounded-[var(--radius-xs)] shadow-sm pointer-events-none transition-opacity ${positions[side]}`}
        >
          {content}
        </div>
      )}
    </div>
  )
}
