# Quantized Game Dialogue Simulator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/k3saj4v4n7/badge.svg)](https://ask.thili.ai/b/k3saj4v4n7)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)

An end-to-end toolkit for game developers to train a custom GPT on game dialogue, apply dynamic and static post-training quantization (PTQ), and simulate inference speedups and memory footprint reductions on resource-constrained devices (e.g. mobile chipsets, handheld consoles, embedded game engines).

---

## What You Are Building

**Goal.** You will build a small, quantized GPT for in-game dialogue, then simulate its memory footprint and generation speed for resource-constrained devices, demonstrating the impact of quantization on model size and simulated performance.

**Done when.** The project is complete when you have trained a small GPT on a custom game dialogue corpus, applied dynamic and static post-training quantization, and measured and compared the original and quantized model sizes and simulated inference speeds.

---

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Prepare a small game dialogue text corpus.**  
      Hint: Focus on cleaning and tokenizing your game dialogue text data.
- [x] **2. Implement a basic GPT architecture and train it on the corpus.**  
      Hint: Use the 'Build GPT from Scratch' drill's techniques to define and train your model.
- [x] **3. Generate sample dialogue from the trained GPT.**  
      Hint: Implement a text generation function using your trained GPT to produce sample dialogue.
- [x] **4. Apply dynamic and static post-training quantization to the model.**  
      Hint: Refer to the 'Shrink a Neural Network with Quantisation' drill for applying dynamic and static post-training quantization.
- [x] **5. Measure and compare the original and quantized model sizes.**  
      Hint: Save your original and quantized models and check their file sizes to measure memory footprint.
- [x] **6. Simulate inference speed by comparing operations on original versus quantized weights.**  
      Hint: Measure the time taken for a fixed number of inference steps for both models, as shown in the 'Shrink a Neural Network with Quantisation' drill.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Game Dialogue Corpus      |
                      |   (NPCs, Quests, Banter)    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  CharTokenizer & Cleaning   |
                      +--------------+--------------+
                                     |
                                     v
        +-------------------------------------------------------+
        |                 MiniGPT Architecture                 |
        |                                                       |
        |  [Token Embedding] + [Positional Embedding]           |
        |                         |                             |
        |                         v                             |
        |          +-----------------------------+              |
        |          | Transformer Block (x2)      |              |
        |          |  - LayerNorm                |              |
        |          |  - Causal Multi-Head Attn   |              |
        |          |  - LayerNorm                |              |
        |          |  - FeedForward (MLP)        |              |
        |          +-----------------------------+              |
        |                         |                             |
        |                         v                             |
        |            [LayerNorm] -> [LM Head]                   |
        +-------------------------+-----------------------------+
                                  |
            +---------------------+---------------------+
            |                                           |
            v                                           v
+-----------------------+                   +-----------------------+
| Dynamic Quantization  |                   |  Static Quantization  |
| (torch.qint8 weights) |                   | (PTQ with Calibration)|
+-----------+-----------+                   +-----------+-----------+
            |                                           |
            +---------------------+---------------------+
                                  |
                                  v
             +-----------------------------------------+
             |   Size & Hardware Inference Simulator   |
             |   - File footprint / Memory budget      |
             |   - Edge device bandwidth speedup       |
             +-----------------------------------------+
```

---

## Quantization Benchmark Results

Benchmarks evaluated using PyTorch on CPU with INT8 dynamic & static post-training quantization:

| Model Variant | Saved Size | Reduction | Compression Ratio | Edge Device Speedup (Simulated) |
| :--- | :---: | :---: | :---: | :---: |
| **Original FP32** | `267.40 KB` | *Baseline (0%)* | $1.00\times$ | $1.00\times$ |
| **Dynamic PTQ (INT8)** | `104.42 KB` | **-60.95%** | **$2.56\times$** | **$2.56\times$** |
| **Static PTQ (INT8)** | `128.64 KB` | **-51.89%** | **$2.08\times$** | **$2.08\times$** |

### Insights for Resource-Constrained Gaming Devices
- **Memory Footprint:** Quantization achieves over **60% memory savings**, fitting easily into strict VRAM/RAM budgets for microcontrollers and mobile titles.
- **Bandwidth Speedup:** On edge devices (Switch, Steam Deck, mobile GPUs), generative LLM token latency is heavily bounded by memory bandwidth. Shrinking model weights to 8-bit enables up to **$2.56\times$ higher token generation throughput**.

---

## Installation & Setup

### Requirements
- Python 3.12+ (supports Python 3.13)
- PyTorch 2.0+
- `git`

### Quickstart

```bash
# 1. Clone repository
git clone https://github.com/eMohan07/quantized-game-dialogue-simulator.git
cd quantized-game-dialogue-simulator

# 2. Set up virtual environment
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment configuration
cp .env.example .env
```

---

## CLI Usage

The simulator provides dedicated subcommands corresponding to each milestone:

```bash
# Verify setup
python -m app.cli hello

# Milestone 1: Inspect cleaned corpus and vocabulary
python -m app.cli corpus

# Milestone 2: Train MiniGPT model
python -m app.cli train

# Milestone 3: Generate in-game dialogue
python -m app.cli generate --prompt "[Guard] "
python -m app.cli generate --prompt "[Hero] "

# Milestone 5: Measure and compare model sizes
python -m app.cli compare-sizes

# Milestone 6: Simulate inference speed & edge device bandwidth speedups
python -m app.cli simulate-speed
```

---

## Running Tests

All milestone implementations are backed by automated tests:

```bash
pytest
```

Output:
```text
tests\test_core.py ......                                                [ 66%]
tests\test_smoke.py ...                                                  [100%]
======================= 9 passed in ~30s =======================
```

---

## Project Structure

```text
quantized-game-dialogue-simulator/
├── app/
│   ├── __init__.py
│   ├── cli.py             # Command-line interface with milestone commands
│   ├── core.py            # Tokenizer, MiniGPT, training, PTQ, size & speed simulation
│   ├── llm.py             # LLM helper utilities
│   └── storage.py         # Key-value store
├── tests/
│   ├── test_core.py       # Comprehensive unit tests for Milestones 1 to 6
│   └── test_smoke.py      # Plumbing and sanity checks
├── .env.example           # Example environment template
├── .gitignore             # Git ignore patterns
├── pytest.ini             # Pytest configuration
├── requirements.txt       # Dependencies (torch, pytest, pandas, etc.)
└── README.md              # Project documentation
```

---

## License

MIT License. Developed for in-game dialog simulation and quantization research.
