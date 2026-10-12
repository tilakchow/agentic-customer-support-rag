import React, { useState, useEffect } from "react"
import { Database, Search, ArrowRight, CheckCircle2, AlertCircle, Layers, FileText } from "lucide-react"
import { fetchKBInfo, searchKB, KBInfoResponse, KBSearchResult } from "../../services/api"

interface KBViewProps {
  onAskInChat: (query: string) => void
}

export const KBView: React.FC<KBViewProps> = ({ onAskInChat }) => {
  const [kbInfo, setKbInfo] = useState<KBInfoResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState("")
  const [searchResults, setSearchResults] = useState<KBSearchResult[]>([])
  const [searching, setSearching] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchKBInfo()
      .then((data) => setKbInfo(data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault()
    if (!searchQuery.trim()) return

    setSearching(true)
    setError(null)
    try {
      const data = await searchKB(searchQuery.trim())
      setSearchResults(data.results)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to search Knowledge Base"
      setError(msg)
    } finally {
      setSearching(false)
    }
  }

  return (
    <div className="flex-1 overflow-y-auto px-6 py-6 max-w-4xl mx-auto w-full select-none">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6 pb-4 border-b border-slate-200">
        <div className="w-10 h-10 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center border border-blue-100 shadow-2xs">
          <Database className="w-5 h-5 text-blue-600" />
        </div>
        <div>
          <h2 className="text-lg font-bold text-slate-900 tracking-tight">
            Knowledge Base Inspector
          </h2>
          <p className="text-xs text-slate-500">
            Real-time Pinecone vector index and dataset configuration
          </p>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 rounded-xl bg-red-50 border border-red-200 text-red-700 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0 text-red-600" />
          <span>{error}</span>
        </div>
      )}

      {/* Pinecone Index Info Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        <div className="p-3.5 rounded-xl border border-slate-200 bg-white shadow-2xs">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">Index Name</span>
          <span className="text-xs font-semibold text-slate-900 mt-1 block truncate">
            {kbInfo?.index_name || "industry-agentic-rag-kb"}
          </span>
        </div>
        <div className="p-3.5 rounded-xl border border-slate-200 bg-white shadow-2xs">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">Namespace</span>
          <span className="text-xs font-semibold text-slate-900 mt-1 block truncate">
            {kbInfo?.namespace || "bitext-support"}
          </span>
        </div>
        <div className="p-3.5 rounded-xl border border-slate-200 bg-white shadow-2xs">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">Dimension</span>
          <span className="text-xs font-semibold text-slate-900 mt-1 block">
            {kbInfo?.dimension || 384} dims (Normalized)
          </span>
        </div>
        <div className="p-3.5 rounded-xl border border-slate-200 bg-white shadow-2xs">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">Status</span>
          <span className="text-xs font-semibold text-emerald-600 mt-1 flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
            Connected & Ready
          </span>
        </div>
      </div>

      {/* Dataset Details Card */}
      <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-2xs mb-6 text-xs text-slate-700">
        <h3 className="font-semibold text-slate-900 mb-2 flex items-center gap-2 text-xs">
          <FileText className="w-4 h-4 text-slate-500" />
          Ingested Training Dataset
        </h3>
        <p className="text-slate-600 leading-relaxed">
          Source: <span className="font-mono text-slate-800">bitext/Bitext-customer-support-llm-chatbot-training-dataset</span>.
          Trained on real e-commerce and customer service procedures with embeddings computed via <span className="font-mono text-slate-800">sentence-transformers/all-MiniLM-L6-v2</span>.
        </p>
      </div>

      {/* Semantic Vector Search Tester */}
      <div className="p-5 rounded-2xl border border-slate-200 bg-white shadow-2xs mb-6">
        <h3 className="text-xs font-bold text-slate-900 tracking-tight mb-1 flex items-center gap-2">
          <Search className="w-4 h-4 text-blue-600" />
          Test Pinecone Vector Retrieval
        </h3>
        <p className="text-[11px] text-slate-500 mb-3">
          Type a test question below to retrieve the exact raw text chunks the LangGraph agent reads from Pinecone.
        </p>

        <form onSubmit={handleSearch} className="flex gap-2">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="e.g. cancel order, refund steps, login problem..."
            className="flex-1 rounded-xl border border-slate-300 px-3 py-2 text-xs text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500"
          />
          <button
            type="submit"
            disabled={searching || !searchQuery.trim()}
            className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:opacity-40 text-white text-xs font-medium cursor-pointer transition-colors shrink-0"
          >
            {searching ? "Searching..." : "Retrieve Chunks"}
          </button>
        </form>

        {/* Retrieved Chunks Display */}
        {searchResults.length > 0 && (
          <div className="mt-4 space-y-3">
            <span className="text-[11px] font-semibold text-slate-600">
              Retrieved {searchResults.length} Chunks:
            </span>
            {searchResults.map((res, i) => (
              <div key={i} className="p-3 rounded-lg border border-slate-100 bg-slate-50/70 text-xs">
                <div className="flex items-center justify-between mb-1.5 pb-1 border-b border-slate-200/50">
                  <span className="font-semibold text-blue-700 text-[11px]">Chunk #{i + 1}</span>
                  <button
                    onClick={() => onAskInChat(searchQuery)}
                    className="text-[11px] text-blue-600 hover:underline flex items-center gap-1 cursor-pointer"
                  >
                    Ask Agent in Chat <ArrowRight className="w-3 h-3" />
                  </button>
                </div>
                <p className="whitespace-pre-wrap text-slate-700 leading-relaxed font-sans text-[11.5px]">
                  {res.content}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
