![llmbenchmarkscom](https://cronitor.io/badges/G8yp5e/production/VnmBXHNorcpEyvbg9ASvxeGp8zU.svg)

# LLM Benchmarks

A comprehensive framework for benchmarking LLM inference speeds across various models and frameworks.

## Overview

This project provides tools to benchmark Large Language Model (LLM) inference speeds across different frameworks, model sizes, and quantization methods. The benchmarks are designed to run both locally and in cloud environments, with results displayed on a dashboard at [llm-benchmarks.com](https://llm-benchmarks.com).

The system uses Docker with various frameworks (vLLM, Transformers, Text-Generation-Inference, llama-cpp) to automate benchmarks and upload results to a MongoDB database. Most frameworks fetch models from the HuggingFace Hub and cache them for on-demand loading, with the exception of llama-cpp/GGUF which requires specially compiled model formats.

## Repository Structure

- **`/api`**: Core benchmarking logic and API clients for different frameworks
- **`/cloud`**: Configuration and Docker setup for cloud-based benchmarks (OpenAI, Anthropic, etc.)
- **`/local`**: Configuration and Docker setup for local benchmarks (Hugging Face, vLLM, GGUF)
  - **`/local/huggingface`**: Transformers and Text-Generation-Inference benchmarks
  - **`/local/vllm`**: vLLM benchmarks
  - **`/local/gguf`**: GGUF/llama-cpp benchmarks
- **`/scripts`**: Utility scripts and notebooks
- **`/static`**: Static assets like benchmark result images
- **`models_config.yaml`**: Configuration for model groups used in benchmarks

## Getting Started

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- Python 3.9+
- MongoDB (optional, for result storage)
- Redis (for task queuing)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/cipher982/llm-benchmarks.git
   cd llm-benchmarks
   ```

2. Set up environment variables:
   ```bash
   # For local benchmarks
   cp local/.env.example local/.env
   # For cloud benchmarks
   cp cloud/.env.example cloud/.env
   ```

3. Edit the `.env` files with your configuration:
   - Set `HF_HUB_CACHE` to your Hugging Face model cache directory
   - Configure MongoDB connection if using (`MONGODB_URI`, `MONGODB_DB`, etc.)
   - Set API keys for cloud providers if benchmarking them
   - Configure Redis connection details

### Running Benchmarks

#### Local Benchmarks

1. Start the local benchmark containers:
   ```bash
   cd local
   docker compose -f docker-compose.local.yml up --build
   ```

2. Run benchmarks for specific frameworks:

   - Hugging Face Transformers:
     ```bash
     python api/run_hf.py --framework transformers --limit 5 --max-size-billion 10 --run-always
     ```

   - Hugging Face Text-Generation-Inference:
     ```bash
     python api/run_hf.py --framework hf-tgi --limit 5 --max-size-billion 10 --run-always
     ```

   - vLLM:
     ```bash
     python api/run_vllm.py --framework vllm --limit 5 --max-size-billion 10 --run-always
     ```

   - GGUF/llama-cpp:
     ```bash
     python api/run_gguf.py --limit 5 --run-always --log-level DEBUG
     ```

#### Cloud Benchmarks

1. Start the cloud benchmark container:
   ```bash
   cd cloud
   docker compose -f docker-compose.cloud.yml up --build
   ```

2. Run benchmarks for cloud providers:
   ```bash
   python api/run_cloud.py --providers openai
   # Or run all configured providers
   python api/run_cloud.py --providers all
   ```

## Viewing Results

Results can be viewed in several ways:

1. **Dashboard**: Visit [llm-benchmarks.com](https://llm-benchmarks.com) to see the latest benchmark results
2. **Log Files**: Check the `logs/` directory for:
   - `benchmarks_local.log` and `benchmarks_cloud.log`: Text logs with detailed metrics
   - `benchmarks_local.json` and `benchmarks_cloud.json`: JSON-formatted logs
3. **MongoDB**: If configured, results are stored in MongoDB collections

## Benchmark Results

The benchmarks measure inference speed across different models, quantization methods, and output token counts. Results indicate that even the slowest performing combinations still handily beat GPT-4 and almost always match or beat GPT-3.5, sometimes significantly.

### Framework Comparisons

Different frameworks show significant performance variations. For example, GGML with cuBLAS significantly outperforms Hugging Face Transformers with BitsAndBytes quantization:

![GGML v HF](https://github.com/cipher982/llm-benchmarks/blob/main/static/ggml-hf-llama-compare.png?raw=true)

### Model Size and Quantization Impact

Benchmarks show how model size and quantization affect inference speed:

#### LLaMA Models
![LLaMA Models](https://github.com/cipher982/llm-benchmarks/blob/main/static/llama_compare_size_and_quant_inference.png?raw=true)

#### Dolly-2 Models
![Dolly2 Models](https://github.com/cipher982/llm-benchmarks/blob/main/static/dolly2_compare_size_and_quant_inference.png?raw=true)

#### Falcon Models
![Falcon Models](https://github.com/cipher982/llm-benchmarks/blob/main/static/falcon_compare_quantization_inference.png?raw=true)

## Hardware Considerations

Benchmarks have been run on various GPUs including:
- NVIDIA RTX 3090
- NVIDIA A10
- NVIDIA A100
- NVIDIA H100

The H100 consistently delivers the fastest performance but at a higher cost (~$2.40/hour). Surprisingly, the A10 performed below expectations despite its higher tensor core count, possibly due to memory bandwidth limitations.

## Contributing

Contributions are welcome! To add new models or frameworks:

1. Fork the repository
2. Create a feature branch
3. Add your implementation
4. Submit a pull request

For more details, see the individual README files in the `/local` and `/cloud` directories.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
