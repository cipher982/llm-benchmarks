# Anti-Complexity Monitoring & Alerting Redesign

**Date:** 2026-08-19  
**Status:** Approved & Synthesized (Reviewed by Hatch Codex Luna & Cursor Grok)  
**Author:** AI Agent  
**Context:** LLM Benchmarks (`llm-benchmarks.com`) monitoring and deadman switch reliability

---

## 1. Problem Statement & Incident Context

Over the past 48 hours, multiple alert cycles paged `alerts@drose.io`:
1. **9:23 AM (Operational Invariant Failure):** `endpoint_targets_are_being_measured` failed due to desynchronized state between `bench_jobs` and `bench_model_health` for retired/budget-exhausted endpoints.
2. **1:37 PM (Deadman Provider Floor False Alarm):** After the endpoint invariant was resolved, a new alert fired for `only 4 providers published, floor is 5`. This occurred because `llm_bench_deadman.py` held a hardcoded legacy threshold (`MIN_PUBLISHED_PROVIDERS = 5`) from before the August 17 OpenRouter consolidation (which reduced direct catalogue lanes to 4: `bedrock`, `openai`, `vertex`, `openrouter`).

### Core Root Causes
- **Conflating Invariants with Product Audits:** Reasoning models (e.g. DeepSeek R1, Claude 3.7 Thinking) exhaust the 64-token budget before outputting visible tokens. They run fine under the 2048-token shadow profile, but cannot publish on the 64-token Delivered TPS chart. Registering `models_measurable_by_the_published_profile` as a failing invariant (`ok: false`) forced every consumer (`watchdog.ts`, `llm_bench_deadman.py`, Sauron `invariant_watch.py`) to maintain custom ignore lists (`audit_invariants`).
- **Deadman Scope Creep:** The deadman switch on `cube` checked business logic and catalogue cardinality with hardcoded integers (`MIN_PUBLISHED_PROVIDERS = 5`, `MIN_ENABLED_MODELS = 100`, etc.) instead of purely monitoring host liveness, DB connectivity, and data flow.
- **Monolithic Alert Fingerprinting:** `llm_bench_deadman.py` hashed all active problems into a single SHA-256 string. Resolving problem A changed the hash from `hash([A, B])` to `hash([B])`, immediately firing a new alert for problem B even if problem B was already alerted.

---

## 2. Refined Architectural Design

```mermaid
graph TD
    subgraph Clifford ["Clifford (Monitored Host)"]
        Runner[Benchmark Daemon] -->|evaluates| Checker[invariants.py]
        Checker -->|pages=true -> results[]| Ledger[bench_check_runs]
        Checker -->|pages=false -> audits[]| Ledger
        Ledger --> WatchdogAPI["/api/watchdog (Next.js)"]
        Ledger --> SauronJob[Sauron invariant_watch.py]
    end
    subgraph Cube ["Cube (Pure Deadman Switch)"]
        Deadman[llm_bench_deadman.py] -->|checks liveness & freshness| WatchdogAPI
        Deadman -->|checks payload presence| PublicAPI["/api/processed?days=3"]
        Deadman -->|per-issue cooldown alert| Email[Gmail SMTP]
    end
```

### Component 1: Invariant vs Audit via Single Flag (`pages: bool`)
- In `llm_bench/ops/invariants.py`, `Invariant` dataclass adds `pages: bool = True`.
- `models_measurable_by_the_published_profile` is marked `pages=False`.
- `record_check_run()` partitions results:
  - `pages=True` entries go to `results` array with `ok: bool`.
  - `pages=False` entries go to `audits` array with `count: int`, `subjects: list[str]`, `description: str` (no `ok: false`).
- `/api/watchdog` already filters `failing: results.filter(r => r.ok === false)`. With audits written to `audits`, `failing` is empty whenever all operational invariants pass.
- All downstream `audit_invariants` ignore sets in Cube and Sauron are deleted.

### Component 2: Pure Deadman Switch on Cube
- Remove all hardcoded cardinality thresholds (`MIN_ENABLED_MODELS`, `MIN_ENABLED_CATALOGUE_PROVIDERS`, `MIN_PUBLISHED_PROVIDERS`, `MIN_PUBLISHED_ROWS`).
- Stop inspecting `invariants.failing` on Cube (Sauron owns operational invariant alerting directly from MongoDB). Cube only alerts if the invariant checker *stops running* (`invariants.age_seconds > 5400`).
- Cube deadman checks strictly:
  1. Watchdog reachable and returns parseable JSON (accepts 200 and 503-with-body).
  2. MongoDB reachable (`mongo.reachable == true`).
  3. Invariant checker running (`invariants.age_seconds <= 5400`).
  4. Benchmark execution moving (`benchmarks.age_seconds <= 10800`).
  5. Public API reachable and returns a non-empty `table` list.

### Component 3: Per-Issue Lifecycle Alert Tracking
- Replace `_fingerprint(problems)` with stable issue keys:
  - `watchdog-unreachable`
  - `mongo-unreachable`
  - `invariants-unread`
  - `invariants-never-run`
  - `invariants-stale`
  - `benchmarks-unread`
  - `benchmarks-never-run`
  - `benchmarks-stale`
  - `public-unreachable`
  - `public-not-json`
  - `public-empty`
- Persistent state file `~/.local/state/llm-bench-deadman.json` stores `{issue_key: {"first_seen": ISO, "last_alerted": ISO}}`.
- An issue only triggers an email if `now - last_alerted >= 6 hours` or if it is newly observed.
- Updating `last_alerted` stamps only the keys actually included in the email.
- Cleared issues are gracefully dropped without altering the cooldown of surviving issues.

### Component 4: Per-Invariant Alerting in Sauron
- In `sauron/jobs/jobs/llm_benchmarks/invariant_watch.py`, track alerts per failing invariant name using `alert_key = f"llm-bench:invariants:{result['name']}"`.
- Resolving one invariant clears that specific alert key without resetting the cooldown for other active invariant failures.

---

## 3. Deployment & Cutover Sequence

1. **Step 1:** Update `llm-benchmarks/api/llm_bench/ops/invariants.py` (`pages` flag, `record_check_run` partitioning). Run test suite.
2. **Step 2:** Update dashboard watchdog tests if needed. Verify `/api/watchdog` output.
3. **Step 3:** Deploy refactored `llm_bench_deadman.py` to `cube`.
4. **Step 4:** Update `sauron/jobs/jobs/llm_benchmarks/invariant_watch.py` for per-invariant alerting.
5. **Step 5:** End-to-end verification across Clifford, Cube, and Sauron.
