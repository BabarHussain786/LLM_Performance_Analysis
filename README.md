# LLM Performance Analysis

This project explores the optimization of **Large Language Models (LLMs)** for **CPU-based inference**. We benchmark and analyze four popular LLMs using various optimization techniques like **ONNX Runtime**, **Quantized ONNX**, **OpenVINO**, and **INT8 quantization** in PyTorch. The focus of the study is to reduce **inference latency**, improve **memory usage**, and **compress model size** for real-time, resource-efficient deployment.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Models Analyzed](#models-analyzed)
3. [Optimization Techniques](#optimization-techniques)
4. [Benchmarking Protocol](#benchmarking-protocol)
5. [Results](#results)
6. [Getting Started](#getting-started)
7. [How to Use](#how-to-use)
8. [Contributing](#contributing)
9. [License](#license)

## Project Overview

Large Language Models have revolutionized fields like Natural Language Processing (NLP), but their computational cost poses challenges, especially for deployment on CPUs. This project aims to optimize LLM performance by leveraging various optimization techniques to improve their efficiency during inference on CPU-based systems.

### Key Features:
- **Benchmarking**: Comparison of multiple LLMs using different optimization techniques.
- **Performance Metrics**: Focus on inference latency, memory footprint, and model compression ratio.
- **Optimization Methods**: Use of ONNX Runtime, Quantized ONNX, OpenVINO, and INT8 quantization.

## Models Analyzed

This study evaluates the following LLMs:
1. **distilbert-base-uncased**: A distilled version of the BERT model (66 million parameters).
2. **distilgpt2**: A smaller, faster version of GPT-2 (82 million parameters).
3. **EleutherAI/gpt-neo-125M**: A GPT-style open-source model (125 million parameters).
4. **microsoft/phi-1.5**: A large-scale model trained on "textbook-quality" data (1.5 billion parameters).

## Optimization Techniques

The following optimization techniques were employed to enhance the performance of the models:
1. **ONNX Runtime**: A framework-independent runtime optimized for machine learning models.
2. **Quantized ONNX**: Reduces model size and accelerates inference by quantizing model weights to lower precision (e.g., INT8).
3. **OpenVINO**: Intel’s toolkit for optimizing deep learning models to run efficiently on Intel hardware.
4. **INT8 Quantization**: Reduces the precision of model weights to INT8 for faster computation on compatible CPUs.

## Benchmarking Protocol

Each model and optimization technique was evaluated under controlled conditions:
- **Warm-up phase**: To ensure that measurements are stable.
- **Metrics measured**:
  - **Inference Latency**: Time taken for the model to make a prediction.
  - **Memory Footprint**: Amount of memory required to load and execute the model.
  - **Compression Ratio**: The reduction in model size after optimization.

We ran the models under different configurations and measured the speed and memory impact of each optimization.

## Results

### Key Findings:
- **Latency Improvements**: Optimization techniques achieved speedups ranging from **1.8x to 4.7x**.
- **Memory Usage**: Quantization and other optimizations significantly reduced memory usage.
- **Model Compression**: Optimization methods reduced model sizes by over **1.5x**.

For detailed results, including charts and numerical analysis, refer to the Jupyter notebook in the repository.

## Getting Started

### Prerequisites
To run the code and replicate the benchmarking results, make sure you have the following installed:

1. **Python 3.x**: Download and install from [python.org](https://www.python.org/downloads/).
2. **Jupyter Notebook**: Install using the following:
   ```bash
   pip install notebook
