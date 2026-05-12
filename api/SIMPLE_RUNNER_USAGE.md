# Simple Benchmark Runner Usage

`bench_simple_runner.py` is a lightweight benchmark runner designed for remote environments without MongoDB access (e.g., EC2 instances).
Production Bedrock uses the dynamic `/runner-config` control plane; `BENCHMARK_MODELS` is only a manual/local fallback unless `BENCHMARK_MODELS_OVERRIDE=1` is set.

## Features

- No MongoDB dependencies
- Posts results to HTTP ingest API
- Supports daemon mode for continuous running
- Configurable via the HTTPS runner-config endpoint, CLI arguments, or environment variables
- Comprehensive validation and error handling

## Prerequisites

1. Install dependencies:
```bash
uv sync
```

2. Set required environment variables:
```bash
export INGEST_API_URL="https://your-ingest-api.com/ingest"
export INGEST_API_KEY="your-api-key"
```

3. Configure provider credentials (e.g., for Bedrock):
```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-east-1"
```

## Usage

### Production Daemon Mode

Fetch the enabled model worklist and per-model request metadata from
`bench-ingest` each cycle:

```bash
export RUNNER_CONFIG_URL="https://bench-ingest.drose.io/runner-config?provider=bedrock"
export RUNNER_CONFIG_TOKEN="..."
export INGEST_API_URL="https://bench-ingest.drose.io/ingest"
export INGEST_API_KEY="..."

uv run python api/bench_simple_runner.py \
  --provider bedrock \
  --daemon
```

This is the normal Bedrock path. Updating MongoDB `models.enabled` or
`models.deprecated` changes the next fetched worklist; the EC2 runner does not
connect to MongoDB directly.

### Single Run Mode (Default)

Run benchmarks once and exit:

```bash
# Using CLI arguments
uv run python api/bench_simple_runner.py \
  --provider bedrock \
  --models "us.anthropic.claude-opus-4-5-20251101-v1:0,amazon.nova-pro-v1:0"

# Manual/local fallback using environment variable
export BENCHMARK_MODELS="us.anthropic.claude-opus-4-5-20251101-v1:0,amazon.nova-pro-v1:0"
uv run python api/bench_simple_runner.py --provider bedrock
```

### Daemon Mode

Run continuously with a specified interval:

```bash
# Run every 30 minutes (default) using RUNNER_CONFIG_URL when present
uv run python api/bench_simple_runner.py --provider bedrock --daemon

# Run every 15 minutes
uv run python api/bench_simple_runner.py \
  --provider bedrock \
  --models "us.anthropic.claude-opus-4-5-20251101-v1:0" \
  --daemon \
  --interval 15
```

### Debug Mode

Enable verbose logging:

```bash
uv run python api/bench_simple_runner.py \
  --provider bedrock \
  --models "us.anthropic.claude-opus-4-5-20251101-v1:0" \
  --debug
```

## Docker Deployment

Example Dockerfile for EC2 deployment:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY api/ ./api/

# Install dependencies
RUN uv sync

# Set environment variables (or use docker-compose env_file)
ENV INGEST_API_URL="https://ingest.example.com/ingest"
ENV INGEST_API_KEY="your-key"
ENV RUNNER_CONFIG_URL="https://bench-ingest.drose.io/runner-config?provider=bedrock"
ENV RUNNER_CONFIG_TOKEN="your-config-token"

# Run in daemon mode
CMD ["uv", "run", "python", "api/bench_simple_runner.py", \
     "--provider", "bedrock", \
     "--daemon", \
     "--interval", "30"]
```

Example docker-compose.yml:

```yaml
version: '3.8'
services:
  bedrock-runner:
    build: .
    environment:
      - INGEST_API_URL=https://ingest.example.com/ingest
      - INGEST_API_KEY=${INGEST_API_KEY}
      - RUNNER_CONFIG_URL=https://bench-ingest.drose.io/runner-config?provider=bedrock
      - RUNNER_CONFIG_TOKEN=${RUNNER_CONFIG_TOKEN}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - AWS_REGION=us-east-1
    restart: unless-stopped
```

## CLI Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--provider` | Yes | - | Provider name (e.g., bedrock, openai, anthropic) |
| `--models` | No* | - | Comma-separated model IDs |
| `--daemon` | No | False | Run continuously in daemon mode |
| `--interval` | No | 30 | Interval in minutes between runs (daemon mode only) |
| `--debug` | No | False | Enable debug logging |

\* Required unless `RUNNER_CONFIG_URL` is set, or `BENCHMARK_MODELS` is set for manual/local fallback.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `INGEST_API_URL` | Yes | URL of the ingest API endpoint |
| `INGEST_API_KEY` | Yes | API key for authentication |
| `RUNNER_CONFIG_URL` | Production | HTTPS control-plane endpoint returning enabled models |
| `RUNNER_CONFIG_TOKEN` | Production | Bearer token for `RUNNER_CONFIG_URL` |
| `BENCHMARK_MODELS` | No | Comma-separated model IDs for manual/local fallback |
| `BENCHMARK_MODELS_OVERRIDE` | No | Set to `1` only for emergency static override |

## Output

### Success
```
13:54:23 [INFO] Running benchmark: bedrock:us.anthropic.claude-opus-4-5-20251101-v1:0
13:54:25 [INFO] ✅ Success bedrock:us.anthropic.claude-opus-4-5-20251101-v1:0 (tps=42.67, out=64)
```

### Failure
```
13:54:23 [INFO] Running benchmark: bedrock:bad-model
13:54:24 [ERROR] Benchmark failed for bedrock:bad-model - ValidationException: Invalid model ID
```

## Monitoring

Check exit codes:
- `0` = All benchmarks succeeded
- `1` = One or more benchmarks failed (single-run mode)

In daemon mode, the process runs indefinitely. Monitor logs for errors.

## Testing

Run unit tests:

```bash
uv run python api/test_simple_runner.py
```

Expected output:
```
Running bench_simple_runner tests...

✅ parse_models_arg tests passed
✅ run_single_benchmark mock test passed
✅ run_single_benchmark validation test passed
✅ provider caching test passed

==================================================
All tests passed! ✅
==================================================
```

## Supported Providers

- `bedrock` - AWS Bedrock
- `openai` - OpenAI API
- `anthropic` - Anthropic API
- `vertex` - Google Vertex AI
- `azure` - Azure OpenAI
- `anyscale` - Anyscale Endpoints
- `together` - Together AI
- `openrouter` - OpenRouter
- `runpod` - RunPod
- `fireworks` - Fireworks AI
- `deepinfra` - DeepInfra
- `groq` - Groq
- `databricks` - Databricks
- `lambda` - Lambda Labs
- `cerebras` - Cerebras

Each provider requires its own authentication setup (API keys, credentials, etc.).

## Differences from the scheduler daemon

| Feature | Scheduler daemon | bench_simple_runner.py |
|---------|------------------|------------------------|
| MongoDB | Required | Not used directly |
| Jobs system | `bench_jobs` | No |
| Freshness checks | `bench_model_health` | No |
| Model discovery | From MongoDB | From `/runner-config`; CLI/env var only as fallback |
| Error tracking | MongoDB collections | Logs only; ingest bridge records success health |
| Use case | Production direct-provider scheduler | Remote runners, primarily Bedrock |

## Troubleshooting

### "INGEST_API_URL not set"
Set the required environment variables before running.

### "Unsupported provider: xyz"
Check that the provider name matches one in PROVIDER_MODULES dict.

### "Missing required metric 'tokens_per_second'"
The provider generate() function returned incomplete metrics. Check provider implementation.

### "Failed to post results to ingest API"
- Check INGEST_API_URL is reachable
- Verify INGEST_API_KEY is correct
- Check network connectivity
- Review logs with `--debug` flag
