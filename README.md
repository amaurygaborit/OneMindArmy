# OneMindArmy

**OneMindArmy** is a high-performance, game-agnostic Reinforcement Learning framework designed to train autonomous AI agents from scratch through self-play. 

At its core, the framework separates the learning algorithms and search heuristics from the game rules. It leverages a highly optimized C++ engine for massive parallelization and a Python orchestrator for continuous integration, making it scalable, modular, and ready for Transformer-based architectures.

## The Game-Agnostic Framework

OneMindArmy is built to understand *any* board game by relying on a strict abstraction layer and a universal data model:

* **Universal State Representation:** Environments are modeled using primitive entities (`Atom`, `Fact`, `Action`) mapped to unified bitsets (`BitsetT`). This natively supports both perfect information games (Dirac delta positions) and imperfect information games (probability clouds).
* **Transformer-Ready Encoding:** The `StateEncoder` serializes the symbolic game state into continuous float tensors using strict absolute positional anchoring and dipole movement representations (penalizing source, boosting destination).
* **Massively Parallel MCTS:** The search relies on a multi-threaded Monte Carlo Tree Search featuring **Gumbel Sequential Halving** for aggressive search space reduction. 
* **GPU Batching & TensorRT:** Neural network inference is decoupled from the tree search. A dedicated `ThreadPool` gathers states across hundreds of parallel games and forwards them in optimized batches to TensorRT-compiled models.
* **Continuous Orchestration:** The training loop is fully automated via `orchestrator.py`, which handles asynchronous data generation, sliding window datasets, hot-reloading of neural weights, and VRAM cleanup.

## Chess Implementation

The first concrete implementation of the OneMindArmy framework is **OMAChess**. By simply satisfying the `ValidGameTraits` C++ concept, the chess module plugs directly into the RL pipeline.

* **Bitboard Engine:** Move generation is fully optimized using magic bitboards for sliding pieces and bitwise operations for pawn pushes and knight jumps.
* **Complete Rule Support:** Full implementation of FIDE rules, including En Passant, castling rights, the 50-move rule, threefold repetition, and insufficient material detection.
* **UCI Protocol Support:** Includes a `UCIHandler` to interact with standard chess GUIs (like Arena or Cute Chess).
* **Integrated Perft Tool:** A built-in performance test tool to validate move generation accuracy and speed against known FEN positions.

---

## Getting Started

### Prerequisites
* **C++20** compatible compiler
* **CMake** (>= 3.18)
* **CUDA Toolkit**
* **TensorRT** (NVIDIA)
* **Python 3.x** (with `ruamel.yaml` and `psutil`)

### Clone & Build

To compile the C++ engine, clone the repository and run CMake. *Note: Replace `<YOUR_CUDA_ARCH>` (e.g., `86` for Ampere, `89` for Ada) and `</path/to/TensorRT>` with your specific hardware and software paths.*

# 1. Clone the repository
```bash
git clone [https://github.com/amaurygaborit/OneMindArmy.git](https://github.com/amaurygaborit/OneMindArmy.git)
```
```bash
cd OneMindArmy
```

# 2. Prepare the build directory
```bash
mkdir build && cd build
```

# 3. Configure CMake (adjust parameters for your GPU)
```bash
cmake -DCMAKE_BUILD_TYPE=Release \\
      -DCMAKE_CUDA_ARCHITECTURES=<YOUR_CUDA_ARCH> \\
      -DTRT_ROOT=</path/to/TensorRT> \\
      ..
```

# 4. Compile the engine
```bash
make -j$(nproc)
```
```bash
cd ..
```

### Run the Training Pipeline
Once the C++ binaries are built, you can launch the autonomous self-play and training loop using the Python orchestrator. The pipeline will read the configuration, generate data, train the network, and dynamically replace the model.

# Launch the orchestrator for 1000 iterations
```bash
python scripts/orchestrator.py configs/chess_train.yaml --iterations 1000
```
