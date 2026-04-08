# Mosquitobrain 

![the ui](https://github.com/Larsby/Mosquitobrain/blob/main/Design/Screenshot%202026-04-06%20at%2007.57.50.png "the ui")


Many years ago I made a VST plugin that used a neural network to try and replicate effects. Sadly the sourcecode was lost in a burglary and I have since not felt the need or the lust to recreate it. Not until vibe coding and VST's became a thing. I downloaded APC the Audio Plugin Coder ( https://github.com/Noizefield/audio-plugin-coder/ ) and made a stab at recreating the plugin. It works but it is mostly filter-like effects, sort of like a combination of eq's. However that might be useful sometimes


# Mosquitobrain User Manual

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
