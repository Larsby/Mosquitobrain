# Mosquitobrain User Manual

Version: v0.0.0

## Formats
- VST3
- Audio Unit (AU)
- Standalone

## Installation (macOS)
1. Copy the `.vst3` bundle to `~/Library/Audio/Plug-Ins/VST3/`.
2. Copy the `.component` bundle to `~/Library/Audio/Plug-Ins/Components/`.
3. Optionally copy `Mosquitobrain.app` to `/Applications/`.

## Core Controls
- Dry/Wet: mix between dry input and model output.
- Train Amount: controls training pass budget.
- Architecture controls: complexity, depth, residual, saturation, recurrence.
- Context controls: past and future context samples.

## File Training
1. Choose Dry WAV and Wet WAV.
2. Press Train.
3. Check progress, ETA, loss, A@1%, and A@5%.

## Diagnostics
Runtime diagnostics show CPU, latency, and active context settings in the editor status line while idle.

## Notes
- The background image is loaded from `plugins/Mosquitobrain/Design/background.png`.
- Replace that PNG with your own 780x700 image to customize the UI background.
