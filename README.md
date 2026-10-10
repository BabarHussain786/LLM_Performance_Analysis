# CPU Inference Optimization for Large Language Models

This project studies CPU-oriented inference optimization for several transformer models. It compares portable runtimes and reduced-precision execution with respect to latency, memory footprint, and model-size reduction.

## Models

The notebook evaluates:

- `distilbert-base-uncased`
- `distilgpt2`
- `EleutherAI/gpt-neo-125M`
- `microsoft/phi-1.5`

## Optimization paths

- ONNX Runtime inference
- quantized ONNX models
- OpenVINO execution
- PyTorch INT8 quantization

## Metrics

The benchmark records inference latency, memory footprint, and model size or compression. The reported speedup range in the original study is 1.8×–4.7×, with model-size reductions above 1.5× for the evaluated configurations. These values are summary ranges; the notebook contains the per-model and per-method observations.

## Benchmark protocol

For each model and runtime, record:

- model revision and conversion/quantization settings;
- CPU model, operating-system version, Python version, and package versions;
- input shape, sequence length, batch size, and thread settings;
- warm-up policy and number of measured repetitions;
- latency statistic and memory-measurement method.

Warm-up runs should be excluded from the reported latency, and optimized results should be compared with a baseline under the same input and threading conditions.

## Installation

```bash
pip install notebook numpy pandas matplotlib scikit-learn torch transformers onnx onnxruntime openvino
```

## Run the notebook

```bash
jupyter notebook "LLM ANALYSIS.ipynb"
```

Execute the notebook from top to bottom after confirming that the required model downloads and conversion tools are available. Generated figures and tables should be retained with the notebook outputs and environment metadata.

## Repository structure

- `LLM ANALYSIS.ipynb` — analysis and benchmark workflow
- `README.md` — experiment scope and reproduction notes

## Interpretation

The results describe the tested models, runtimes, and hardware configuration. They should not be generalized to all transformer architectures or CPU platforms without repeating the measurements under matched conditions.

## License

This repository is provided as research material. Check the repository files for any applicable license and cite the project when reusing its code or measurements.
