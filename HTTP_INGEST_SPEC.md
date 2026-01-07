# HTTP Ingest for Remote Benchmark Runners

**Status:** Phase 4 - Deploy & Test
**Goal:** Bedrock benchmarks running on EC2 get metrics into MongoDB

## Problem

EC2 instance (`ml-tuner-demo-server`) has Bedrock access via IAM but can't reach MongoDB (Tailscale-only). Need a way to get benchmark results from EC2 into the database.

## Solution

HTTP ingest: EC2 POSTs results to an API on clifford, API writes to MongoDB.

## Decision Log

### Decision: Minimal simple runner over dual-mode headless
**Context:** `bench_headless.py` is tightly coupled to MongoDB (jobs, errors, freshness)
**Choice:** Create `bench_simple_runner.py` - a lightweight runner with no MongoDB deps
**Rationale:** Simpler than making headless work in two modes. Reuses existing provider code.
**Revisit if:** Need more sophisticated job management on EC2

### Decision: Reuse existing provider code
**Context:** Could duplicate benchmark logic or import from library
**Choice:** Import `llm_bench.cloud.providers.bedrock.generate()`
**Rationale:** No code duplication, fixes in one place apply everywhere

### Decision: Static model list for EC2
**Context:** Can't query MongoDB for models from EC2
**Choice:** Pass models via env var `BENCHMARK_MODELS` or CLI args
**Rationale:** Simple, explicit, no extra API needed

---

## Phases

### Phase 1: Add HTTP logging to core library
**Files:** `api/llm_bench/http_output.py` (new)

Add `log_http()` function that POSTs to ingest API. Keep it simple:
- Takes config + metrics
- POSTs to `INGEST_API_URL`
- Returns True/False

**Acceptance:**
- [x] `log_http()` exists and can POST a benchmark result
- [x] Uses `httpx` (already in deps)
- [x] Reads URL/key from env vars

**Implementation Notes:**
- Created `api/llm_bench/http_output.py` with `log_http()` function
- Takes CloudConfig instance and metrics dict
- Combines config.to_dict() and metrics into single payload
- POSTs to INGEST_API_URL with X-API-Key header
- Returns True on success, False on any failure
- Includes proper error logging for debugging
- Tested with verification script (scratch/test_http_output.py)

### Phase 2: Create simple runner
**Files:** `api/bench_simple_runner.py` (new), `api/test_simple_runner.py` (new)

Minimal benchmark runner:
- Takes provider + models as args
- Calls `provider.generate()` for each model
- Calls `log_http()` with results
- Loops on interval (daemon mode) or runs once

**Acceptance:**
- [x] Can run: `python api/bench_simple_runner.py --provider bedrock --models "model1,model2"`
- [x] Calls actual provider generate() function
- [x] POSTs results to ingest API via log_http()
- [x] Has --daemon mode for continuous running

**Implementation Notes:**
- Created `api/bench_simple_runner.py` with argparse CLI
- Supports `--provider`, `--models`, `--daemon`, `--interval`, `--debug` flags
- Reads models from CLI args or BENCHMARK_MODELS env var
- Uses same PROVIDER_MODULES mapping and caching as bench_headless.py
- Validates metrics (output_tokens, generate_time, tokens_per_second)
- Single-run mode (default) or daemon mode with configurable interval
- No MongoDB dependencies - only uses http_output.log_http()
- Comprehensive unit tests with mocking (4 tests, all passing)

### Phase 3: Update ingest API
**Files:** `bench-ingest/main.py`

Already mostly done. May need:
- Error endpoint if we want error logging

**Acceptance:**
- [x] `/ingest` accepts results from simple runner
- [x] Data appears in MongoDB

**Implementation Notes:**
- API already complete at `/Users/davidrose/git/llmbench/bench-ingest/`
- Has `/ingest` (single), `/ingest/batch` (multiple), `/health` endpoints
- Error endpoint deferred - simple runner logs errors locally and continues
- Ready for deployment to clifford

### Phase 4: Deploy & test
- Deploy `bench-ingest` to clifford via Coolify
- Deploy simple runner to EC2 via Docker
- Verify end-to-end

**Acceptance:**
- [ ] Bedrock benchmarks appearing in MongoDB
- [ ] Dashboard shows Bedrock data

---

## Out of Scope
- Dual-mode `bench_headless.py`
- `/models` endpoint (static list is fine)
- Batch ingest (single POST per result is fine for now)
- Complex retry/spool logic (log failure and move on)
