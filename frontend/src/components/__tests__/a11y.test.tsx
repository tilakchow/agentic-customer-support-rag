import { describe, it, expect } from "vitest"
import React from "react"
import { render } from "@testing-library/react"
import axe from "axe-core"
import { Composer } from "../Composer"
import { WelcomeHero } from "../WelcomeHero"
import { ChatMessage } from "../ChatMessage"
import { ErrorBanner } from "../ErrorBanner"

describe("Accessibility Audit (axe-core)", () => {
  it("WelcomeHero has zero serious or critical a11y violations", async () => {
    const { container } = render(<WelcomeHero onSelectQuery={() => {}} />)
    const results = await axe.run(container, {
      rules: {
        // Color contrast in jsdom can produce false positives without full layout engine, so focus on ARIA & DOM structure
      },
    })
    const severeViolations = results.violations.filter(
      (v) => v.impact === "serious" || v.impact === "critical"
    )
    expect(severeViolations).toEqual([])
  })

  it("Composer has zero serious or critical a11y violations", async () => {
    const { container } = render(
      <Composer input="" onChange={() => {}} onSend={() => {}} loading={false} />
    )
    const results = await axe.run(container)
    const severeViolations = results.violations.filter(
      (v) => v.impact === "serious" || v.impact === "critical"
    )
    expect(severeViolations).toEqual([])
  })

  it("ChatMessage has zero serious or critical a11y violations", async () => {
    const { container } = render(
      <ChatMessage
        message={{
          id: "msg-1",
          role: "assistant",
          content: "Hello! This is a grounded customer support response.",
          timestamp: "10:30 AM",
        }}
      />
    )
    const results = await axe.run(container)
    const severeViolations = results.violations.filter(
      (v) => v.impact === "serious" || v.impact === "critical"
    )
    expect(severeViolations).toEqual([])
  })

  it("ErrorBanner has zero serious or critical a11y violations", async () => {
    const { container } = render(
      <ErrorBanner message="Network disconnected" onRetry={() => {}} />
    )
    const results = await axe.run(container)
    const severeViolations = results.violations.filter(
      (v) => v.impact === "serious" || v.impact === "critical"
    )
    expect(severeViolations).toEqual([])
  })
})
