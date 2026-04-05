# Style Guide v1

## Visual Direction
Minimal modern, neutral-dark studio panel with a single fresh accent for actionable states.

## Color System
- Background: #0F141A
- Surface: #1E2935
- Surface Alt: #263445
- Accent: #48D0A8
- Accent Dim: #2E8B73
- Text Primary: #E8EEF2
- Text Secondary: #9DB0BF
- Border: #314252

## Typography
- Primary font: Lato / Sans-serif fallback
- Title: 26px semibold
- Section labels: 14px semibold
- Control labels: 13px medium
- Value text: 12px regular

## Spacing Rules
- Outer padding: 28px
- Section gap: 20px
- Control spacing: 36px (horizontal)
- Minimum touch target: 52x52px
- Corner radius:
  - Panels: 12px
  - Knob wells: 16px

## Control Styling
- Rotary knobs use dual-ring style:
  - Outer ring: surface alt
  - Active arc: accent
  - Indicator line: text primary
- Discrete slot knob displays 16 tick marks around circumference.
- Value labels are right below controls with fixed-width numeric alignment.

## States
- Idle: muted surface with subtle border.
- Hover/focus: border brightens to accent dim.
- Active (training action bound later): accent ring increases thickness by 1px.
- Disabled: opacity 60 percent, no accent arc.

## Layout Behavior
- Keep control cluster centered at all times.
- Header and footer stretch full width but remain low visual weight.
- No animated meters in v1.
