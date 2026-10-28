import React from "react"
import {
  Plus,
  MessageSquare,
  Trash2,
  Database,
  Settings,
  Sun,
  Moon,
  PanelLeftClose,
  PanelLeft,
} from "lucide-react"
import { SavedThread } from "./views/HistoryView"
import { IconButton } from "./ui/IconButton"

export interface AppSidebarProps {
  collapsed: boolean
  onToggleCollapse: () => void
  currentThreadId: string
  threads: SavedThread[]
  onSelectThread: (thread: SavedThread) => void
  onDeleteThread: (threadId: string) => void
  onNewChat: () => void
  onOpenKB: () => void
  onOpenSettings: () => void
  theme: "light" | "dark"
  onToggleTheme: () => void
}

export const AppSidebar: React.FC<AppSidebarProps> = ({
  collapsed,
  onToggleCollapse,
  currentThreadId,
  threads,
  onSelectThread,
  onDeleteThread,
  onNewChat,
  onOpenKB,
  onOpenSettings,
  theme,
  onToggleTheme,
}) => {
  return (
    <aside
      className={`h-full bg-[var(--color-surface)] border-r border-[var(--color-border)] flex flex-col transition-all duration-200 select-none z-30 shrink-0 ${
        collapsed ? "w-16" : "w-64"
      }`}
    >
      {/* Sidebar Header: Brand + New Chat */}
      <div className="p-3 border-b border-[var(--color-border-subtle)] flex items-center justify-between">
        {!collapsed ? (
          <div className="flex items-center gap-2 px-1">
            <div className="w-6 h-6 rounded-[var(--radius-xs)] bg-[var(--color-ink)] text-[var(--color-canvas)] flex items-center justify-center font-bold text-[12px]">
              B
            </div>
            <span className="font-semibold text-[14px] text-[var(--color-ink)] tracking-tight">
              Bitext Support
            </span>
          </div>
        ) : null}

        <IconButton
          variant="ghost"
          size="sm"
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          onClick={onToggleCollapse}
          className="mx-auto"
        >
          {collapsed ? (
            <PanelLeft className="w-4 h-4 text-[var(--color-ink-muted)]" />
          ) : (
            <PanelLeftClose className="w-4 h-4 text-[var(--color-ink-muted)]" />
          )}
        </IconButton>
      </div>

      {/* New Thread CTA */}
      <div className="p-2.5">
        <button
          type="button"
          onClick={onNewChat}
          className={`w-full flex items-center gap-2 py-2 px-3 rounded-[var(--radius-md)] bg-[var(--color-canvas)] border border-[var(--color-border)] hover:border-[var(--color-brand)] text-[var(--color-ink)] text-[13px] font-medium transition-all shadow-[var(--shadow-card)] cursor-pointer ${
            collapsed ? "justify-center px-0" : ""
          }`}
          aria-label="Start new conversation"
          title="New Chat"
        >
          <Plus className="w-4 h-4 text-[var(--color-brand)] shrink-0" />
          {!collapsed && <span>New Thread</span>}
        </button>
      </div>

      {/* Thread list grouped by history */}
      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-1">
        {!collapsed && threads.length > 0 && (
          <div className="px-2 py-1 text-[10.5px] font-semibold uppercase tracking-wider text-[var(--color-ink-faint)]">
            Recent Conversations
          </div>
        )}

        {threads.map((t) => {
          const isActive = t.threadId === currentThreadId
          return (
            <div
              key={t.threadId}
              onClick={() => onSelectThread(t)}
              className={`group flex items-center justify-between p-2 rounded-[var(--radius-sm)] transition-colors cursor-pointer text-[12.5px] ${
                isActive
                  ? "bg-[var(--color-surface-hover)] text-[var(--color-ink)] font-medium border border-[var(--color-border)]"
                  : "text-[var(--color-ink-muted)] hover:text-[var(--color-ink)] hover:bg-[var(--color-surface-hover)] border border-transparent"
              } ${collapsed ? "justify-center" : ""}`}
              title={t.title}
            >
              <div className="flex items-center gap-2 min-w-0">
                <MessageSquare className="w-3.5 h-3.5 shrink-0 text-[var(--color-ink-faint)]" />
                {!collapsed && (
                  <span className="truncate max-w-[150px]">{t.title}</span>
                )}
              </div>

              {!collapsed && (
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation()
                    onDeleteThread(t.threadId)
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 text-[var(--color-ink-faint)] hover:text-[var(--color-danger)] transition-all cursor-pointer"
                  aria-label="Delete thread"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          )
        })}

        {!collapsed && threads.length === 0 && (
          <div className="px-3 py-6 text-center text-[12px] text-[var(--color-ink-faint)]">
            No past sessions yet
          </div>
        )}
      </div>

      {/* Sidebar Footer: KB dialog button, Settings dialog button, Theme Toggle */}
      <div className="p-2 border-t border-[var(--color-border-subtle)] space-y-1">
        <button
          type="button"
          onClick={onOpenKB}
          className={`w-full flex items-center gap-2.5 p-2 rounded-[var(--radius-sm)] text-[12.5px] text-[var(--color-ink-muted)] hover:text-[var(--color-ink)] hover:bg-[var(--color-surface-hover)] transition-colors cursor-pointer ${
            collapsed ? "justify-center" : ""
          }`}
          aria-label="Open Knowledge Base inspector"
          title="Knowledge Base"
        >
          <Database className="w-4 h-4 text-[var(--color-brand)] shrink-0" />
          {!collapsed && <span>Knowledge Base</span>}
        </button>

        <button
          type="button"
          onClick={onOpenSettings}
          className={`w-full flex items-center gap-2.5 p-2 rounded-[var(--radius-sm)] text-[12.5px] text-[var(--color-ink-muted)] hover:text-[var(--color-ink)] hover:bg-[var(--color-surface-hover)] transition-colors cursor-pointer ${
            collapsed ? "justify-center" : ""
          }`}
          aria-label="Open System Settings"
          title="Settings"
        >
          <Settings className="w-4 h-4 shrink-0" />
          {!collapsed && <span>System Settings</span>}
        </button>

        <button
          type="button"
          onClick={onToggleTheme}
          className={`w-full flex items-center gap-2.5 p-2 rounded-[var(--radius-sm)] text-[12.5px] text-[var(--color-ink-muted)] hover:text-[var(--color-ink)] hover:bg-[var(--color-surface-hover)] transition-colors cursor-pointer ${
            collapsed ? "justify-center" : ""
          }`}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
        >
          {theme === "dark" ? (
            <Sun className="w-4 h-4 text-amber-400 shrink-0" />
          ) : (
            <Moon className="w-4 h-4 text-[var(--color-ink-muted)] shrink-0" />
          )}
          {!collapsed && (
            <span>{theme === "dark" ? "Light theme" : "Dark theme"}</span>
          )}
        </button>
      </div>
    </aside>
  )
}
