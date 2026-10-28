import React, { useState } from "react"
import { Server, Cpu, Database, RefreshCw, CheckCircle2, AlertCircle } from "lucide-react"
import { checkHealth } from "../../services/api"

interface SettingsViewProps {
  backendOnline: boolean
  onResetSession: () => void
  onClearStorage: () => void
}

export const SettingsView: React.FC<SettingsViewProps> = ({
  backendOnline,
  onResetSession,
  onClearStorage,
}) => {
  const [testingPing, setTestingPing] = useState(false)
  const [pingResult, setPingResult] = useState<string | null>(null)

  const handleTestPing = async () => {
    setTestingPing(true)
    setPingResult(null)
    const start = Date.now()
    try {
      await checkHealth()
      const latency = Date.now() - start
      setPingResult(`Connected successfully (${latency}ms latency)`)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error"
      setPingResult(`Connection failed: ${msg}`)
    } finally {
      setTestingPing(false)
    }
  }

  return (
    <div className="space-y-4 text-[13px] text-[var(--color-ink)] select-none">
      {/* Backend Status Card */}
      <div className="p-4 rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)]">
        <h3 className="text-[13px] font-semibold text-[var(--color-ink)] mb-1 flex items-center gap-2">
          <Server className="w-4 h-4 text-[var(--color-brand)]" />
          <span>FastAPI Service Endpoint</span>
        </h3>
        <p className="text-[12px] text-[var(--color-ink-muted)] mb-3">
          Edge reverse proxy routing with rate limiting and thread token validation.
        </p>

        <div className="flex items-center justify-between p-3 rounded-[var(--radius-md)] bg-[var(--color-surface-hover)] border border-[var(--color-border-subtle)] text-[12px]">
          <div className="flex items-center gap-2.5">
            <span
              className={`w-2 h-2 rounded-full ${
                backendOnline ? "bg-[var(--color-success)]" : "bg-amber-500"
              }`}
            />
            <span className="font-mono text-[var(--color-ink)]">
              {backendOnline ? "Service Online (port 8000)" : "Service Offline"}
            </span>
          </div>

          <button
            type="button"
            onClick={handleTestPing}
            disabled={testingPing}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-[var(--radius-sm)] bg-[var(--color-surface)] border border-[var(--color-border)] hover:bg-[var(--color-surface-hover)] text-[11.5px] font-medium text-[var(--color-ink)] transition-colors cursor-pointer"
          >
            <RefreshCw className={`w-3 h-3 ${testingPing ? "animate-spin" : ""}`} />
            <span>Test Health</span>
          </button>
        </div>

        {pingResult && (
          <div
            className={`mt-2.5 p-2 rounded-[var(--radius-sm)] text-[11.5px] flex items-center gap-2 ${
              pingResult.includes("successfully")
                ? "bg-[var(--color-success-subtle)] text-[var(--color-success)]"
                : "bg-[var(--color-danger-subtle)] text-[var(--color-danger)]"
            }`}
          >
            {pingResult.includes("successfully") ? (
              <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
            ) : (
              <AlertCircle className="w-3.5 h-3.5 shrink-0" />
            )}
            <span>{pingResult}</span>
          </div>
        )}
      </div>

      {/* Architecture Specs */}
      <div className="p-4 rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)]">
        <h3 className="text-[13px] font-semibold text-[var(--color-ink)] mb-3 flex items-center gap-2">
          <Cpu className="w-4 h-4 text-[var(--color-brand)]" />
          <span>Active Architecture</span>
        </h3>

        <div className="grid grid-cols-2 gap-2 text-[12px]">
          <div className="p-2.5 rounded-[var(--radius-sm)] bg-[var(--color-surface-hover)] border border-[var(--color-border-subtle)]">
            <span className="text-[10px] text-[var(--color-ink-faint)] uppercase tracking-wider block">Orchestrator</span>
            <span className="font-medium text-[var(--color-ink)]">LangGraph StateGraph</span>
          </div>
          <div className="p-2.5 rounded-[var(--radius-sm)] bg-[var(--color-surface-hover)] border border-[var(--color-border-subtle)]">
            <span className="text-[10px] text-[var(--color-ink-faint)] uppercase tracking-wider block">Inference</span>
            <span className="font-medium text-[var(--color-ink)]">Groq (gpt-oss-20b)</span>
          </div>
          <div className="p-2.5 rounded-[var(--radius-sm)] bg-[var(--color-surface-hover)] border border-[var(--color-border-subtle)]">
            <span className="text-[10px] text-[var(--color-ink-faint)] uppercase tracking-wider block">Retrieval</span>
            <span className="font-medium text-[var(--color-ink)]">Pinecone (384d cosine)</span>
          </div>
          <div className="p-2.5 rounded-[var(--radius-sm)] bg-[var(--color-surface-hover)] border border-[var(--color-border-subtle)]">
            <span className="text-[10px] text-[var(--color-ink-faint)] uppercase tracking-wider block">Web Fallback</span>
            <span className="font-medium text-[var(--color-ink)]">Tavily Search API</span>
          </div>
        </div>
      </div>

      {/* Storage and Thread Controls */}
      <div className="p-4 rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)]">
        <h3 className="text-[13px] font-semibold text-[var(--color-ink)] mb-1 flex items-center gap-2">
          <Database className="w-4 h-4 text-[var(--color-ink-muted)]" />
          <span>Session State & Checkpoints</span>
        </h3>
        <p className="text-[12px] text-[var(--color-ink-muted)] mb-3">
          Manage local threads saved in browser storage.
        </p>

        <div className="flex gap-2.5">
          <button
            type="button"
            onClick={onResetSession}
            className="px-3 py-1.5 rounded-[var(--radius-sm)] bg-[var(--color-surface-hover)] border border-[var(--color-border)] hover:bg-[var(--color-border)] text-[var(--color-ink)] text-[12px] font-medium transition-colors cursor-pointer"
          >
            Reset Active Thread
          </button>
          <button
            type="button"
            onClick={onClearStorage}
            className="px-3 py-1.5 rounded-[var(--radius-sm)] bg-[var(--color-danger-subtle)] border border-[var(--color-danger-border)] text-[var(--color-danger)] text-[12px] font-medium hover:opacity-90 transition-opacity cursor-pointer"
          >
            Clear All History
          </button>
        </div>
      </div>
    </div>
  )
}
