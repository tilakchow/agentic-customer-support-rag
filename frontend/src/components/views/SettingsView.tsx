import React, { useState } from "react"
import { Settings, Server, Cpu, Database, RefreshCw, CheckCircle2, AlertCircle } from "lucide-react"
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
    <div className="flex-1 overflow-y-auto px-6 py-6 max-w-4xl mx-auto w-full select-none">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6 pb-4 border-b border-slate-200">
        <div className="w-10 h-10 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center border border-blue-100 shadow-2xs">
          <Settings className="w-5 h-5 text-blue-600" />
        </div>
        <div>
          <h2 className="text-lg font-bold text-slate-900 tracking-tight">
            System & Architecture Settings
          </h2>
          <p className="text-xs text-slate-500">
            Backend connection parameters, model configurations, and local session management
          </p>
        </div>
      </div>

      {/* Backend Connection Card */}
      <div className="p-5 rounded-2xl border border-slate-200 bg-white shadow-2xs mb-6">
        <h3 className="text-xs font-bold text-slate-900 tracking-tight mb-2 flex items-center gap-2">
          <Server className="w-4 h-4 text-blue-600" />
          FastAPI Backend Service
        </h3>
        <p className="text-[11.5px] text-slate-500 mb-4 leading-relaxed">
          The frontend connects to the local FastAPI backend running with Uvicorn.
        </p>

        <div className="flex items-center justify-between p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-xs">
          <div className="flex items-center gap-3">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                backendOnline ? "bg-emerald-500 ring-2 ring-emerald-100" : "bg-amber-500"
              }`}
            />
            <div>
              <span className="font-semibold text-slate-800">API Endpoint: </span>
              <span className="font-mono text-slate-600">http://localhost:8000</span>
            </div>
          </div>

          <button
            onClick={handleTestPing}
            disabled={testingPing}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 text-xs font-medium cursor-pointer transition-colors shadow-2xs"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${testingPing ? "animate-spin" : ""}`} />
            <span>Test Ping</span>
          </button>
        </div>

        {pingResult && (
          <div
            className={`mt-3 p-2.5 rounded-lg text-xs flex items-center gap-2 ${
              pingResult.includes("successfully")
                ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                : "bg-red-50 text-red-700 border border-red-200"
            }`}
          >
            {pingResult.includes("successfully") ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
            ) : (
              <AlertCircle className="w-4 h-4 text-red-600 shrink-0" />
            )}
            <span>{pingResult}</span>
          </div>
        )}
      </div>

      {/* Model & Agent Specs Grid */}
      <div className="p-5 rounded-2xl border border-slate-200 bg-white shadow-2xs mb-6">
        <h3 className="text-xs font-bold text-slate-900 tracking-tight mb-3 flex items-center gap-2">
          <Cpu className="w-4 h-4 text-blue-600" />
          Active Agent Architecture
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
          <div className="p-3 rounded-xl bg-slate-50/70 border border-slate-100">
            <span className="text-[10.5px] font-medium text-slate-400 uppercase tracking-wider block">LLM Engine</span>
            <span className="font-semibold text-slate-800 mt-0.5 block">Groq (openai/gpt-oss-20b)</span>
            <span className="text-[11px] text-slate-500">Ultra-fast inference & structured JSON output</span>
          </div>
          <div className="p-3 rounded-xl bg-slate-50/70 border border-slate-100">
            <span className="text-[10.5px] font-medium text-slate-400 uppercase tracking-wider block">Orchestrator</span>
            <span className="font-semibold text-slate-800 mt-0.5 block">LangGraph StateGraph</span>
            <span className="text-[11px] text-slate-500">Dynamic routing, evidence grading & retry loops</span>
          </div>
          <div className="p-3 rounded-xl bg-slate-50/70 border border-slate-100">
            <span className="text-[10.5px] font-medium text-slate-400 uppercase tracking-wider block">Memory Engine</span>
            <span className="font-semibold text-slate-800 mt-0.5 block">LangGraph MemorySaver</span>
            <span className="text-[11px] text-slate-500">Thread ID conversational state persistence</span>
          </div>
          <div className="p-3 rounded-xl bg-slate-50/70 border border-slate-100">
            <span className="text-[10.5px] font-medium text-slate-400 uppercase tracking-wider block">Web Fallback</span>
            <span className="font-semibold text-slate-800 mt-0.5 block">Tavily Search API</span>
            <span className="text-[11px] text-slate-500">Real-time web search when KB evidence is weak</span>
          </div>
        </div>
      </div>

      {/* Storage & Reset Actions */}
      <div className="p-5 rounded-2xl border border-slate-200 bg-white shadow-2xs">
        <h3 className="text-xs font-bold text-slate-900 tracking-tight mb-2 flex items-center gap-2">
          <Database className="w-4 h-4 text-slate-700" />
          Data & Cache Management
        </h3>
        <p className="text-[11.5px] text-slate-500 mb-4 leading-relaxed">
          Manage stored threads and session checkpoints in local storage.
        </p>

        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={onResetSession}
            className="px-3.5 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-medium cursor-pointer transition-colors"
          >
            Reset Active Session ID
          </button>
          <button
            onClick={onClearStorage}
            className="px-3.5 py-2 rounded-xl border border-red-200 text-red-600 hover:bg-red-50 text-xs font-medium cursor-pointer transition-colors"
          >
            Clear All History & Cache
          </button>
        </div>
      </div>
    </div>
  )
}
