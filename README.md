# Quantized Game Dialogue Simulator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/k3saj4v4n7/badge.svg)](https://ask.thili.ai/b/k3saj4v4n7)

This project helps game developers build a small, quantized GPT for in-game dialogue. You will train a custom GPT on game script text, then apply quantization techniques to reduce its size and simulate its efficiency for resource-constrained devices. The goal is to understand how quantization impacts model footprint and generation speed.

## What you are building

**Goal.** You will build a small, quantized GPT for in-game dialogue, then simulate its memory footprint and generation speed for resource-constrained devices, demonstrating the impact of quantization on model size and simulated performance.

**Done when.** The project is complete when you have trained a small GPT on a custom game dialogue corpus, applied dynamic and static post-training quantization, and measured and compared the original and quantized model sizes and simulated inference speeds.

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
- [ ] **5. Measure and compare the original and quantized model sizes.**  
      Hint: Save your original and quantized models and check their file sizes to measure memory footprint.
- [ ] **6. Simulate inference speed by comparing operations on original versus quantized weights.**  
      Hint: Measure the time taken for a fixed number of inference steps for both models, as shown in the 'Shrink a Neural Network with Quantisation' drill.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Analyze the impact of quantization on the quality of the generated dialogue, perhaps by having the model generate more diverse samples and evaluating them.
- If the drill provides options, experiment with different quantization bit-widths or configurations beyond the basic dynamic/static examples.
- Create simple visualizations (e.g., bar charts) to clearly present the comparison of model sizes and simulated inference speeds.
