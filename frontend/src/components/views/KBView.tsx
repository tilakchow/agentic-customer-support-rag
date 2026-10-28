import React, { useState, useEffect } from "react"
import { Search, ArrowRight, CheckCircle2, AlertCircle, FileText, Loader2 } from "lucide-react"
import { fetchKBInfo, searchKB, KBInfoResponse, KBSearchResult } from "../../services/api"

interface KBViewProps {
  onAskInChat: (query: string) => void
}

export const KBView: React.FC<KBViewProps> = ({ onAskInChat }) => {
  const [kbInfo, setKbInfo] = useState<KBInfoResponse | null>(null)
  const [_loading, setLoading] = useState(true)
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
    <div className="space-y-4 text-[13px] text-[var(--color-ink)] select-none">
      {error && (
        <div className="p-3 rounded-[var(--radius-md)] bg-[var(--color-danger-subtle)] border border-[var(--color-danger-border)] text-[var(--color-danger)] text-[12px] flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0 text-[var(--color-danger)]" />
          <span>{error}</span>
        </div>
      )}

      {/* Index Parameters Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
        <div className="p-3 rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface-hover)]">
          <span className="text-[10px] font-semibold text-[var(--color-ink-faint)] uppercase tracking-wider block">
            Index Name
          </span>
          <span className="text-[12.5px] font-medium text-[var(--color-ink)] mt-0.5 block truncate">
            {kbInfo?.index_name || "customer-support-rag"}
          </span>
        </div>

        <div className="p-3 rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface-hover)]">
          <span className="text-[10px] font-semibold text-[var(--color-ink-faint)] uppercase tracking-wider block">
            Namespace
          </span>
          <span className="text-[12.5px] font-medium text-[var(--color-ink)] mt-0.5 block truncate">
            {kbInfo?.namespace || "bitext-support"}
          </span>
        </div>

        <div className="p-3 rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface-hover)]">
          <span className="text-[10px] font-semibold text-[var(--color-ink-faint)] uppercase tracking-wider block">
            Dimensions
          </span>
          <span className="text-[12.5px] font-medium text-[var(--color-ink)] mt-0.5 block">
            {kbInfo?.dimension || 384} (MiniLM)
          </span>
        </div>

        <div className="p-3 rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface-hover)]">
          <span className="text-[10px] font-semibold text-[var(--color-ink-faint)] uppercase tracking-wider block">
            Status
          </span>
          <span className="text-[12px] font-medium text-[var(--color-success)] mt-0.5 flex items-center gap-1">
            <CheckCircle2 className="w-3.5 h-3.5 text-[var(--color-success)]" />
            <span>Ready</span>
          </span>
        </div>
      </div>

      {/* Dataset Attribution */}
      <div className="p-3.5 rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface-hover)] text-[12px]">
        <div className="flex items-center gap-2 font-medium text-[var(--color-ink)] mb-1">
          <FileText className="w-3.5 h-3.5 text-[var(--color-brand)]" />
          <span>Ingested Dataset: Bitext Customer Support</span>
        </div>
        <p className="text-[var(--color-ink-muted)] leading-relaxed">
          Embeddings pre-indexed with <code className="font-mono text-[11px] text-[var(--color-ink)]">sentence-transformers/all-MiniLM-L6-v2</code> for semantic customer service policy search.
        </p>
      </div>

      {/* Semantic Search Sandbox */}
      <div className="p-4 rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] shadow-xs">
        <h3 className="text-[13px] font-semibold text-[var(--color-ink)] mb-1 flex items-center gap-2">
          <Search className="w-3.5 h-3.5 text-[var(--color-brand)]" />
          <span>Query Pinecone Index</span>
        </h3>
        <p className="text-[11.5px] text-[var(--color-ink-muted)] mb-3">
          Retrieve raw chunk embeddings that feed into the LangGraph evidence grader.
        </p>

        <form onSubmit={handleSearch} className="flex gap-2">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="e.g. refund timeframe, invoice lookup, cancel shipment..."
            className="flex-1 rounded-[var(--radius-sm)] border border-[var(--color-border)] px-3 py-2 text-[12.5px] text-[var(--color-ink)] placeholder-[var(--color-ink-faint)] bg-[var(--color-surface)] focus:outline-none focus:border-[var(--color-brand)]"
          />
          <button
            type="submit"
            disabled={searching || !searchQuery.trim()}
            className="px-3.5 py-2 rounded-[var(--radius-sm)] bg-[var(--color-ink)] text-[var(--color-canvas)] text-[12px] font-medium hover:opacity-90 disabled:opacity-40 transition-opacity cursor-pointer shrink-0"
          >
            {searching ? (
              <span className="flex items-center gap-1.5">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Searching...</span>
              </span>
            ) : (
              "Search Index"
            )}
          </button>
        </form>

        {searchResults.length > 0 && (
          <div className="mt-3.5 space-y-2 max-h-64 overflow-y-auto pr-1">
            {searchResults.map((res, i) => (
              <div
                key={i}
                className="p-3 rounded-[var(--radius-sm)] border border-[var(--color-border-subtle)] bg-[var(--color-surface-hover)] text-[12px]"
              >
                <div className="flex items-center justify-between pb-1.5 mb-1.5 border-b border-[var(--color-border-subtle)]">
                  <span className="font-semibold text-[var(--color-brand)] text-[11px]">
                    Retrieved Document #{i + 1}
                  </span>
                  <button
                    type="button"
                    onClick={() => onAskInChat(searchQuery)}
                    className="text-[11px] text-[var(--color-brand)] hover:underline flex items-center gap-1 cursor-pointer font-medium"
                  >
                    <span>Ask in chat</span>
                    <ArrowRight className="w-3 h-3" />
                  </button>
                </div>
                <p className="text-[var(--color-ink-muted)] leading-relaxed text-[11.5px] whitespace-pre-wrap">
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
