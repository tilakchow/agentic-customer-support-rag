import React from "react"
import { ArrowRight } from "lucide-react"

export interface WelcomeHeroProps {
  onSelectQuery: (query: string) => void
}

interface ExamplePrompt {
  title: string
  query: string
  category: string
}

export const WelcomeHero: React.FC<WelcomeHeroProps> = ({ onSelectQuery }) => {
  // Real customer support questions grounded in the Bitext Knowledge Base
  const REAL_KB_EXAMPLES: ExamplePrompt[] = [
    {
      title: "Refund Request Window",
      query: "How long do I have to request a refund on an item?",
      category: "Returns & Policy",
    },
    {
      title: "Order Cancellation Steps",
      query: "Can I cancel my order if it has already been processed?",
      category: "Order Management",
    },
    {
      title: "Update Delivery Address",
      query: "How do I change my shipping address for a pending shipment?",
      category: "Shipping & Delivery",
    },
    {
      title: "Invoicing and Receipts",
      query: "Where can I view and download past tax invoices?",
      category: "Billing & Invoicing",
    },
  ]

  return (
    <div className="w-full max-w-2xl mx-auto flex flex-col items-center text-center py-10 px-4 select-none">
      {/* Product Mark & Wordmark */}
      <div className="flex items-center gap-2.5 mb-3">
        <div className="w-9 h-9 rounded-[var(--radius-md)] bg-[var(--color-ink)] text-[var(--color-canvas)] flex items-center justify-center font-bold text-[16px] tracking-tight shadow-[var(--shadow-card)]">
          B
        </div>
        <span className="font-semibold text-[20px] tracking-tight text-[var(--color-ink)]">
          Bitext Support
        </span>
      </div>

      <h1 className="text-[26px] sm:text-[30px] font-semibold tracking-tight text-[var(--color-ink)] mb-2">
        Where support meets precision search
      </h1>
      <p className="text-[14px] text-[var(--color-ink-muted)] max-w-md mb-8 leading-relaxed">
        Grounded answers retrieved directly from internal policy documents with real-time web verification.
      </p>

      {/* Suggested genuine questions grid */}
      <div className="w-full grid grid-cols-1 sm:grid-cols-2 gap-3 text-left">
        {REAL_KB_EXAMPLES.map((item, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => onSelectQuery(item.query)}
            className="p-3.5 rounded-[var(--radius-xl)] bg-[var(--color-surface)] border border-[var(--color-border)] hover:border-[var(--color-brand)] hover:shadow-[var(--shadow-elevation)] transition-all cursor-pointer group flex flex-col justify-between"
          >
            <div>
              <span className="text-[10.5px] uppercase tracking-wider font-semibold text-[var(--color-brand)] block mb-1">
                {item.category}
              </span>
              <p className="text-[13px] font-medium text-[var(--color-ink)] leading-snug group-hover:text-[var(--color-brand)] transition-colors">
                {item.query}
              </p>
            </div>
            <div className="flex items-center justify-end mt-2 text-[var(--color-ink-faint)] group-hover:text-[var(--color-brand)]">
              <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
