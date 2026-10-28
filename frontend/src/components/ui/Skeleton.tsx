import React from "react"

export interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  width?: string | number
  height?: string | number
}

export const Skeleton: React.FC<SkeletonProps> = ({
  width,
  height,
  className = "",
  style,
  ...props
}) => {
  return (
    <div
      aria-hidden="true"
      className={`animate-pulse bg-[var(--color-surface-hover)] rounded-[var(--radius-sm)] ${className}`}
      style={{
        width,
        height,
        ...style,
      }}
      {...props}
    />
  )
}
