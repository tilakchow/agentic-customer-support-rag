# Frontend Design System

> **Identity:** *Bitext Support* — Scholar's parchment behind clean glass. Precision customer support search & reasoning assistant powered by Bitext knowledge base.

## 1. Design Language & Principles

- **Warm Paper Desk Canvas:** Avoid sterile pure white (`#ffffff`) canvases. Light mode uses a calm, organic parchment tone (`#faf8f5`) with deep warm-ink text (`#201e18`) and hairline warm borders (`#e6e2da`).
- **Singular Restrained Accent:** A single deep teal (`#016a71` in light mode, `#0c8c95` in dark mode) is reserved strictly for active navigation states, selected chips, and focus boundaries. No competing second chromatic colors.
- **Calm, Low Elevation:** Shadows are strictly minimal (1px subtle cards, soft elevation on modal dialogs). Surfaces feel like distinct sheets resting on desk parchment.
- **Self-Hosted Clean Typography:** Powered by `Inter Variable` with modest weights (400–500) and relaxed line heights. No heavy or distracting display fonts.
- **No Emojis in UI Chrome:** Chrome icons use clean SVG glyphs (`lucide-react`) at uniform stroke weight (1.5–1.75). No emoji in headers, buttons, cards, or status indicators.

---

## 2. Token Definitions (`src/index.css`)

### Color System

| Token | Light Value | Dark Value | Purpose |
| :--- | :--- | :--- | :--- |
| `--color-canvas` | `#faf8f5` | `#131311` | Application base background canvas |
| `--color-surface` | `#ffffff` | `#1c1b18` | Card surfaces, modals, popovers |
| `--color-surface-hover` | `#f5f2ec` | `#262420` | Hover states on cards, buttons, nav items |
| `--color-border` | `#e6e2da` | `#2e2c27` | Structural hairlines and input borders |
| `--color-border-subtle` | `#eeebe5` | `#24221d` | Divider lines and card separators |
| `--color-ink` | `#201e18` | `#f4f2ed` | Primary reading text and titles |
| `--color-ink-muted` | `#6e6b63` | `#a3a096` | Secondary helper labels, timestamps |
| `--color-ink-faint` | `#96938b` | `#737067` | Placeholders, inactive icons |
| `--color-brand` | `#016a71` | `#0c8c95` | Active nav states, brand accent, focus rings |
| `--color-danger` | `#b91c1c` | `#ef4444` | Error banners, destructive actions |
| `--color-success` | `#15803d` | `#22c55e` | Status indicators, copy confirmations |

### Spacing & Sizing Scale

- **Base Unit:** 4px
- **Scale:**
  - `4px` (`spacing-1`): Micro gaps, badges
  - `8px` (`spacing-2`): Icon-to-label gaps, tight lists
  - `12px` (`spacing-3`): Card internal padding, button paddings
  - `16px` (`spacing-4`): Section gutters, standard card padding
  - `24px` (`spacing-6`): Modal padding, conversational message spacing
  - `32px` (`spacing-8`): Large section dividers

### Radii Hierarchy

- **Inputs & Buttons:** `var(--radius-sm)` (6px) or `var(--radius-md)` (10px)
- **Cards & Reading Blocks:** `var(--radius-xl)` (16px)
- **Pills & Mode Chips:** `var(--radius-full)` (9999px)

---

## 3. Component Primitives (`src/components/ui`)

1. **`Button`**: Primary (ink fill), Secondary (surface border), Ghost, Danger. Includes focus rings and accessible disabled states.
2. **`IconButton`**: Requires `aria-label` for screen reader compliance; touch target minimum 36px–44px.
3. **`Input` & `Textarea`**: Auto-styled with brand focus indicators, optional icons, and error states.
4. **`Chip`**: Pill action item for suggestions or status selection with active teal fill.
5. **`Card`**: 16px radius, hairline border, flat elevation.
6. **`Tooltip`**: Keyboard accessible (via `:focus`) and hover tooltips for compact icon controls.
7. **`Dialog`**: Modal with backdrop blur, `Escape` key dismissal, outside click dismissal, and `role="dialog"`.
8. **`Skeleton`**: Low-contrast pulse placeholder for asynchronous loading.
9. **`Toast`**: Status notification primitive for alerts and confirmations.

---

## 4. Do's and Don'ts

### Do
- Always use CSS variables (`var(--color-...)`) for colors to ensure automatic light/dark mode support.
- Ensure all interactive icon buttons have an explicit `aria-label`.
- Maintain reading column maximum width at `900px` for optimal eye travel and comprehension.
- Respect `prefers-reduced-motion` and `prefers-color-scheme`.

### Don't
- Don't introduce arbitrary saturated primary colors (no bright blues, purples, or greens).
- Don't use emojis in navigation chrome, action cards, or button labels.
- Don't build controls that have no real backend backing (no fake upload buttons or non-functional dropdown pills).
- Don't use font weights above 500 for normal text.
