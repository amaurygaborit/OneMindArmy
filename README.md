# OneMindArmy

**OneMindArmy** is a high-performance, general-purpose Reinforcement Learning framework written in **C++20**.

Designed to be game-agnostic, it unifies **Perfect Information** games (Chess, Go) and **Imperfect Information** games (Poker, Stratego) under a single, highly optimized architecture. It draws inspiration from **AlphaZero** for search dynamics and **ReBeL** for handling belief states via Bayesian updates.

## Key Features

### Unified Framework for Any Game
OneMindArmy is not just a chess engine; it is a framework. Adding a new game requires implementing only a few standardized interfaces, regardless of the game's complexity or information visibility.

- **Perfect Information**: Standard MCTS with PUCT (AlphaZero style).
- **Imperfect Information** *(In Progress)*:
  - **Probabilistic Belief States**: Facts are represented as probability distributions rather than absolute values.
  - **Bayesian Updates**: Beliefs evolve deterministically based on actions and observations, replacing naive sampling.
  - **ReBeL-like Search**: MCTS operates over the *Belief Space* rather than the raw state space.

### High-Performance Architecture
- **Asynchronous Parallel MCTS**: Lock-free tree search with `std::atomic`, designed for massive scalability on multi-core CPUs.
- **Batched Inference**: Opportunistic batching system that aggregates leaf evaluations from multiple threads to maximize GPU throughput via **CUDA**.
- **Virtual Loss (Lc0 Style)**: Optimistic parallelization ensuring diverse exploration without thread collision.
- **Tree Reusage**: Configurable persistence of the search tree between moves to save computation.
- **Zero-Copy Design**: Extensive use of Memory Pools and `AlignedVec` to minimize allocation overhead.

### Modular & Configurable
- **Hierarchical Configuration**: Complete separation between Hardware (`backend`), Algorithm (`engine`), and Gameplay (`session`) settings via YAML.
- **Fast Terminal Detection**: CPU-based "fast-path" for terminal states (Checkmate/Fold) bypasses the Neural Network for instant responses.

## System Architecture

The framework revolves around a **GameTypeRegistry** that injects dependencies into a highly optimized worker pool.

1.  **MCTSThreadPool**: Orchestrates three types of workers:
    - **Gather Threads**: Traverse the tree, apply Virtual Loss, and queue nodes for evaluation.
    - **Inference Threads**: Batch requests and execute the `NeuralNet` forward pass on GPU.
    - **Backprop Threads**: Update node statistics ($N, W, Q$) and handle belief updates.
2.  **The Engine**: Pure C++ logic defining the game rules.
3.  **The Handler**: Manages the game loop, switching between training, self-play, or human interaction.

## Installation

### Prerequisites
- **C++20 Compiler** (MSVC 2022 recommended on Windows)
- **CMake** (≥ 3.20)
- **NVIDIA CUDA Toolkit** (Required for NeuralNet inference)

### Build Instructions

1. Clone the repository
```bash
git clone https://github.com/amaurygaborit/OneMindArmy.git
```
```bash
cd OneMindArmy
```

2. Clone external dependencies (YAML-CPP, etc.)
```bash
git clone https://github.com/jbeder/yaml-cpp.git external/yaml-cpp
```

3. Configure and Build
```bash
mkdir build && cd build
```
```bash
cmake ..
```
```bash
cmake --build . --config Release
```

## Configuration
The framework uses a unified config.yaml to manage everything from GPU allocation to game strategy.

## How to Add a New Game
To add a game (e.g., Poker), implement the following 4 interfaces:

**ITraits**: Define your State and Action structures (Facts tensor) and constants (NumPlayers, ActionSpace, ...).
**IEngine**: Implement getValidActions, applyAction, and isTerminal.
**IRenderer**: Visualization logic (Console, GUI, or JSON output).
**IRequester**: How to get input for human players.

Register your game in CMakeLists.txt
