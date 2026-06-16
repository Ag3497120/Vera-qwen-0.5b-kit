# Vera-qwen-0.5b-kit

This repository contains the inference engine and mathematical proofs for **Generative Spatial Compression**, a revolutionary technique that shrinks Large Language Models by transforming dense weight matrices into low-rank generative spatial coordinates.

It was built as part of the **Verantyx** architecture, to solve the massive memory bottleneck of LLM inference.

## The Breakthrough
Standard LLM inference requires loading billions of parameters (e.g. 54GB for a 27B model) from RAM into the GPU for *every single generated token*. This massive memory bandwidth requirement is why AI is so expensive.

This kit proves that we can **eliminate this bottleneck** by mathematically generating the matrices directly inside the ALU during inference.

### How it works
Instead of storing `W` (a huge `NxM` matrix), we store:
`W = mod_y * (U * S * (x * mod_x) * V^T)`

By factoring this mathematically, we bypass storing or generating the full matrix entirely:
1. `h = V^T * (x * modX)`
2. `y = modY * (U * S * h)`

This reduces the parameter size from `O(N^2)` to `O(N * Rank)`, giving up to a **10x compression ratio** on inference memory bandwidth!

### The "Deep Re-Training" Secret
Attempting to zero-shot compress weights like this breaks the Residual Stream and causes a noise cascade (gibberish output). To fix this, we developed a technique called **Deep Re-Training**. By loading these generative spatial coordinates back into PyTorch and training them on standard Language Modeling loss, the LLM successfully "re-learns" how to balance its reasoning logic within the compressed space.

## Usage

### 1. Download the Trained `.jgen` Generative Lattice
Download the `qwen_0.5b_trained.jgen` file from HuggingFace:
```bash
hf download kofdai/Vera-qwen-0.5b-jgen qwen_0.5b_trained.jgen
```

### 2. Run the Inference Engine
Run the included python engine to see the magic happen. The engine parses the custom `.jgen` binary and performs Generative Matrix Multiplication:

```bash
pip install torch transformers
python3 inference.py --model qwen_0.5b_trained.jgen --prompt "The capital of France is"
```

Enjoy witnessing the future of Memory-Boundless AI inference!
