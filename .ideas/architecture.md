# DSP Architecture Specification

## Core Components
- Input capture router: receives live input and routes audio to both inference path and optional training capture buffers.
- Training example buffers: lock-free ring buffers for source and target recordings used by offline training jobs.
- Feature encoder: converts audio blocks to compact feature vectors for model processing.
- Neural inference engine: compact feed-forward model running in real time on the audio thread.
- Offline trainer worker: background thread that fits model weights from captured input/output examples.
- Model store (16 slots): persistent storage and recall for trained models and metadata.
- Dry/wet mixer: blends processed output with original input.
- Output safety stage: limiter/clip guard to prevent overload after neural processing.

## Processing Chain
Input -> Input Capture Router -> Feature Encoder -> Neural Inference Engine -> Output Safety Stage -> Dry/Wet Mixer -> Output

### Control-Side Training Flow
Capture Input/Target -> Training Example Buffers -> Offline Trainer Worker -> Model Store Slot (1-16) -> Active Model Swap for Inference

## Parameter Mapping

| Parameter | Component | Function | Range |
|-----------|-----------|----------|-------|
| dry_wet | Dry/Wet Mixer | Crossfades original and neural-processed signal | 0.0 to 1.0 |
| train_amount | Offline Trainer Worker | Scales training intensity (epoch budget / update depth) for offline retraining jobs | 0.0 to 1.0 |
| model_slot | Model Store | Selects active preset slot for model load/save/recall | 1 to 16 |

## Complexity Assessment
Score: 5
Rationale: Mosquitobrain combines offline model fitting and real-time neural inference under hard real-time constraints. It requires strict audio-thread safety, asynchronous training orchestration, model lifecycle persistence, and robust state transitions for train/retrain workflows, which places it in research-level complexity.
