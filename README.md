# OneMindArmy

OneMindArmy is a C++ framework for implementing multi-player, perfect- and imperfect-information games with a minimal API, featuring MCTS (PUCT), Transformer-based policy/value modeling, belief-state estimation, and automated self-play training.

## Key Features

- **Minimal Game API**: Implement only a few interfaces:
  - `ITrait` : game-specific traits and state structure
  - `IEngine` : game rules, state transitions, valid actions
  - `IRenderer` : optional visualization
  - `IRequester` : optional input interface
  - `IHandler` : optional hooks for AI or custom logic
- **Monte-Carlo Tree Search (PUCT)**:
  - Optimizes action selection using policy, value, and belief probabilities.
  - Handles observed states and predicts hidden states as seen from the perspective of other players.
- **Transformer-based Neural Model**:
  - Converts state (elements, meta, last action) into token sequences.
  - Aggregates past states into history input for a Transformer-Encoder.
  - `BeliefHead` predicts each player’s observed state.
  - `PolicyHead` and `ValueHead` produce policy distributions and per-player values.
- **Belief-aware MCTS Expansion**:
  - Samples next-player observed states during node expansion.
  - Integrates predicted beliefs into planning.
- **Training Pipeline**:
  - Fully automated self-play.
  - Learns policy, value, and belief networks from simulations.

## Prerequisites

- **Windows + Visual Studio 2022** (C++20 + CUDA workload)
- **CMake** (≥ 3.20)
- **NVIDIA CUDA Toolkit** (GPU required for inference and training)
  - Download from NVIDIA’s official site and install compatible drivers.

## Getting Started

Clone the main repository and its dependencies:
```bash
git clone https://github.com/<your-username>/OneMindArmy.git
```
```bash
cd OneMindArmy
```
```bash
git clone --depth 1 https://github.com/jbeder/yaml-cpp.git external/yaml-cpp
```
git clone --depth 1 https://github.com/jbeder/yaml-cpp.git external/yaml-cpp
