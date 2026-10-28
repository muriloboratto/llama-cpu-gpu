# llama.cpp — CPU/GPU Offloading, GGUF Quantization & Inference Benchmark

A reproducible, notebook-based experiment for evaluating **LLM inference on heterogeneous CPU + NVIDIA GPU systems** using [llama.cpp](https://github.com/ggml-org/llama.cpp). The project compares CPU-only execution, partial GPU layer offloading, and maximum GPU offloading, with optional GGUF quantization and measurements of execution time and memory usage.

## Objectives

- Investigate **CPU/GPU workload placement** using llama.cpp's `-ngl` option.
- Compare quantized **GGUF** model variants, when provided.
- Measure inference **decode throughput**, **prefill throughput**, **wall-clock latency**, **peak host RSS**, and **approximate incremental GPU memory**.
- Save per-run logs and structured CSV/JSON outputs for subsequent analysis.
- Visualize comparisons across offloading configurations.

## Notebook

**File:** `Llama_CPP_CPU_GPU_Offloading_GGUF_Benchmark.ipynb`

Open [Google Colab](https://colab.research.google.com/drive/1Hw_SH05pZdHl4J_Fmqjs2uRmCpctPlM0?usp=sharing), upload the notebook, then choose **Runtime → Change runtime type → GPU**. Run the cells in order.


## Requirements

- Linux or Google Colab (Linux runtime).
- Python 3, Git, CMake, a C++ compiler, and Jupyter/Colab.
- `psutil`, `pandas`, and `matplotlib` (the notebook installs `psutil` if missing; install the others if necessary).
- Optional: NVIDIA GPU, CUDA toolkit (`nvcc`), and `nvidia-smi` for CUDA offloading and GPU-memory sampling.
- A locally accessible GGUF model, or a **direct download URL** for one.

The notebook selects a CUDA build when both `nvcc` and `nvidia-smi` are available; otherwise, it builds for CPU. GPU visibility alone does not guarantee that a CUDA build can be compiled.

## Quick start

1. Open `Llama_CPP_CPU_GPU_Offloading_GGUF_Benchmark.ipynb` in Jupyter or Google Colab.
2. In Colab, select a GPU runtime if you want to test GPU offloading.
3. Run the **Clone and build llama.cpp** cells. The notebook clones the upstream repository and builds `llama-cli` and `llama-quantize`.
4. Configure the GGUF model, prompt, and benchmark settings in the **Configure model and prompt** section.
5. Optionally enable quantization or add existing GGUF variants.
6. Run the benchmark and results/plotting sections in order.

### Default model and parameters in the supplied notebook

```python
MODEL_PATH = ROOT / 'qwen2.5-0.5b-instruct-q8_0.gguf'
MODEL_URL = 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf'
PROMPT = 'Explain CUDA programming and heterogeneous CPU-GPU workload optimization in 5 sentences.'
N_PREDICT = 128
CPU_THREADS = min(8, os.cpu_count() or 4)
CONTEXT_SIZE = 2048
REPEATS = 3
```

The default model is **Qwen2.5-0.5B-Instruct Q8_0** (approximately 0.629 GiB on disk in the recorded run). Model availability and licensing should be checked before use.

## CPU/GPU offloading configurations

| Configuration | `-ngl` | Meaning |
|---|---:|---|
| `CPU` | `0` | No transformer-layer GPU offloading requested. |
| `Hybrid_8_layers` | `8` | Request up to eight layers on the GPU. |
| `Max_GPU` | `999` | Request maximum supported GPU offloading, subject to model/backend/VRAM constraints. |

The notebook tests all three configurations when `USE_CUDA=True`; otherwise, it tests CPU-only. It runs configurations sequentially, with three repetitions by default.

Equivalent standalone commands (adjust model path):

```bash
# CPU-only
./llama.cpp/build-cuda/bin/llama-cli -m model.gguf -ngl 0 -p 'Explain CUDA' -n 128

# Partial offloading
./llama.cpp/build-cuda/bin/llama-cli -m model.gguf -ngl 8 -p 'Explain CUDA' -n 128

# Maximum requested offloading
./llama.cpp/build-cuda/bin/llama-cli -m model.gguf -ngl 999 -p 'Explain CUDA' -n 128
```

For CPU-only builds, use `build-cpu/bin/llama-cli` instead. The commands above are illustrative; the notebook also controls context size, CPU threads, and warmup behavior.

## Optional GGUF quantization

Quantization is **disabled by default**:

```python
RUN_QUANTIZATION = False
SOURCE_F16_GGUF = ROOT / 'model-f16.gguf'
QUANT_TYPES = ['Q4_K_M', 'Q8_0']
EXTRA_MODELS = {}
```

To generate variants, provide a compatible **F16/BF16 GGUF** source and set `RUN_QUANTIZATION=True`. The notebook invokes `llama-quantize` and adds generated models to the benchmark. Alternatively, populate `EXTRA_MODELS` with paths to existing quantized GGUF files.

**Important:** Quantization can reduce memory requirements but may affect model quality. The notebook measures performance and memory, **not answer quality**. Do not treat Q4_K_M and Q8_0 as quality-equivalent without separate evaluation.

## Collected metrics

| Output field | Description | Interpretation / caveat |
|---|---|---|
| `decode_tokens_per_second` | Token-generation rate parsed from llama.cpp timing output | Decode throughput; not total end-to-end tokens/s. |
| `prompt_tokens_per_second` | Prompt-processing rate parsed from llama.cpp timings | Prefill throughput. |
| `wall_latency_s` | Time from process launch to exit | Includes process startup and model loading. |
| `peak_host_rss_mib` | Maximum sampled RSS of the process and its children | Approximate process resident RAM, not total system RAM. |
| `peak_gpu_delta_mib` | Maximum sampled GPU memory minus idle baseline | Approximate, GPU-wide, **not per-process**; other workloads distort results. |
| `model_size_gib` | GGUF file size on disk | Not the runtime memory footprint. |
| `returncode` | Process exit status | Nonzero runs are retained for diagnosis. |

Memory is sampled periodically (approximately every 100 ms), so short-lived peaks may be missed. `nvidia-smi` is optional; GPU-memory metrics may be unavailable on CPU-only systems.

### Known issue in the supplied run

The recorded notebook execution reports **`decode tok/s: None` in every run**, despite successful process exit. This means the timing parser did not extract a valid decode rate from the llama.cpp output for that version. **Do not report tokens/s or infer them from these runs without fixing/validating the parser against the actual logs.** Inspect the `*.log` files and adjust `timing_metrics()` to match the installed llama.cpp timing format. The notebook's throughput charts will be absent or empty until valid timing values are captured.

## Recorded example: wall-clock latency

The supplied notebook includes the following successful run times for its default Q8_0 model:

| Placement | Run 1 (s) | Run 2 (s) | Run 3 (s) | Median (s) |
|---|---:|---:|---:|---:|
| CPU | 10.249 | 9.699 | 9.669 | 9.699 |
| Hybrid_8_layers | 6.328 | 6.466 | 6.186 | 6.328 |
| Max_GPU | 2.081 | 1.908 | 1.921 | 1.921 |

These are **end-to-end subprocess wall times**, not pure decode latencies. They are illustrative results from the captured environment, not a controlled cross-platform benchmark. The notebook records llama.cpp commit `ac8e169` for this run.

## Results and output files

The notebook writes results to `llama_benchmark_results/`:

```text
llama_benchmark_results/
├── benchmark.csv                 # Per-run measurements
├── benchmark.json                # Per-run measurements
├── summary.csv                   # Median values for successful runs
├── base_CPU_1.log                # Raw stdout/stderr (example)
├── base_Hybrid_8_layers_1.log    # Raw stdout/stderr (example)
├── base_Max_GPU_1.log            # Raw stdout/stderr (example)
├── wall_latency_s.png            # Chart when data are available
├── peak_host_rss_mib.png         # Chart when data are available
└── peak_gpu_delta_mib.png        # Chart when data are available
```

Other plots, including `decode_tokens_per_second.png`, are produced **only when the corresponding metric is available**. Additional log files are generated for each repetition and model.

## Reproducibility recommendations

For a meaningful CPU/GPU comparison:

1. Record the **llama.cpp commit**, GPU model/VRAM, CPU, RAM, CUDA/driver versions, and model checksum.
2. Keep the model, prompt, context size, generation length, CPU thread count, and sampling settings constant.
3. Run one benchmark process at a time and keep the GPU otherwise idle.
4. Use multiple repetitions; distinguish cold-start runs from warm-cache runs.
5. Compare median and variability, and inspect raw logs for failures or missing metrics.
6. For deeper latency analysis, separate **model loading**, **prefill**, **time to first token**, and **decode** instead of relying only on subprocess wall time.
7. When comparing quantization formats, additionally evaluate **output quality** on a fixed task set.
