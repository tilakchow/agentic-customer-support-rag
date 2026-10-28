import { describe, it, expect, vi } from "vitest"
import React from "react"
import { render, screen, fireEvent } from "@testing-library/react"
import { Composer } from "../Composer"
import { ErrorBanner } from "../ErrorBanner"
import { StreamingTimeline } from "../StreamingTimeline"
import { SourcesSection } from "../SourcesSection"

describe("Composer Component", () => {
  it("renders textarea with placeholder and autonomous indicator", () => {
    const handleChange = vi.fn()
    const handleSend = vi.fn()

    render(
      <Composer
        input=""
        onChange={handleChange}
        onSend={handleSend}
        loading={false}
      />
    )

    const textarea = screen.getByLabelText(/ask a customer support question/i)
    expect(textarea).toBeDefined()
    expect(screen.getByText(/pinecone kb \+ tavily/i)).toBeDefined()
  })

  it("submits on Enter key without shift", () => {
    const handleChange = vi.fn()
    const handleSend = vi.fn()

    render(
      <Composer
        input="How do I return an item?"
        onChange={handleChange}
        onSend={handleSend}
        loading={false}
      />
    )

    const textarea = screen.getByLabelText(/ask a customer support question/i)
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: false })
    expect(handleSend).toHaveBeenCalledTimes(1)
  })

  it("does not submit on Shift+Enter", () => {
    const handleChange = vi.fn()
    const handleSend = vi.fn()

    render(
      <Composer
        input="How do I return an item?"
        onChange={handleChange}
        onSend={handleSend}
        loading={false}
      />
    )

    const textarea = screen.getByLabelText(/ask a customer support question/i)
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: true })
    expect(handleSend).not.toHaveBeenCalled()
  })

  it("displays Stop button when streaming", () => {
    const handleStop = vi.fn()

    render(
      <Composer
        input=""
        onChange={vi.fn()}
        onSend={vi.fn()}
        onStop={handleStop}
        loading={true}
      />
    )

    const stopButton = screen.getByLabelText(/stop generating answer/i)
    expect(stopButton).toBeDefined()
    fireEvent.click(stopButton)
    expect(handleStop).toHaveBeenCalledTimes(1)
  })

  it("enforces 1500 character limit warning", () => {
    const longText = "a".repeat(1501)
    render(
      <Composer
        input={longText}
        onChange={vi.fn()}
        onSend={vi.fn()}
        loading={false}
      />
    )

    expect(screen.getByText(/exceeds maximum supported limit of 1500/i)).toBeDefined()
  })
})

describe("ErrorBanner Component", () => {
  it("renders offline error cleanly", () => {
    const handleRetry = vi.fn()
    render(
      <ErrorBanner
        message="Backend offline: Failed to fetch"
        onRetry={handleRetry}
      />
    )

    expect(screen.getByText(/backend unreachable/i)).toBeDefined()
    const retryBtn = screen.getByText(/retry/i)
    fireEvent.click(retryBtn)
    expect(handleRetry).toHaveBeenCalledTimes(1)
  })

  it("renders 429 rate limit warning", () => {
    render(<ErrorBanner message="HTTP 429: Too Many Requests" />)
    expect(screen.getByText(/rate limit reached/i)).toBeDefined()
  })
})

describe("StreamingTimeline Component", () => {
  it("renders active node progress badge", () => {
    render(
      <StreamingTimeline
        activeNode="retrieve_kb"
        visitedNodes={["route_question", "retrieve_kb"]}
        statusMessage="Searching internal knowledge base..."
      />
    )

    expect(screen.getByText(/searching internal knowledge base/i)).toBeDefined()
    expect(screen.getByText(/searching knowledge base/i)).toBeDefined()
  })
})

describe("SourcesSection Component", () => {
  it("renders citation cards when citations array is provided", () => {
    const sources = [
      {
        doc_id: "doc-1",
        intent: "refund_policy",
        score: 0.92,
        content: "Refunds can be requested within 30 days of purchase.",
      },
    ]

    render(<SourcesSection sourceUsed="private_kb" sources={sources} />)
    expect(screen.getByText(/refund policy/i)).toBeDefined()
    expect(screen.getByText(/92% match/i)).toBeDefined()
    expect(screen.getByText(/refunds can be requested within 30 days/i)).toBeDefined()
  })
})
