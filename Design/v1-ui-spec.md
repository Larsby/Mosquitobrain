# UI Specification v1

## Layout
- Window: 800x600px
- Sections:
  - Header bar (plugin title and model slot readout)
  - Center control cluster (three primary controls)
  - Footer status row (training state text only)
- Grid:
  - 12-column logical grid
  - Header spans full width (row 1)
  - Center cluster occupies columns 3-10 (rows 2-5)
  - Footer spans full width (row 6)

## Controls
| Parameter | Type | Position | Range | Default |
|-----------|------|----------|-------|---------|
| dry_wet | Rotary knob | Center cluster left | 0.0 - 1.0 | 0.5 |
| train_amount | Rotary knob | Center cluster middle | 0.0 - 1.0 | 0.7 |
| model_slot | Rotary selector knob | Center cluster right | 1 - 16 | 1 |

## Color Palette
- Background: #0F141A
- Primary: #1E2935
- Accent: #48D0A8
- Text: #E8EEF2

## Style Notes
- Minimal modern visual language with restrained contrast and clear typography.
- Centered composition with strong whitespace and no metering in v1.
- Controls prioritize readability and deterministic interaction over ornament.
- Slot control uses stepped visual ticks to indicate discrete values.
