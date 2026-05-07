Scripts to help prepare MongoDB for the benchmark scheduler.

**Note:** For adding models to production, use `../manage-models.sh` from the parent directory. These scripts are for staging/testing or emergency operations.

Prereqs
- Install `mongosh` on your machine.
- Set env vars pointing to your staging DB/collections.

Env vars
- `MONGODB_URI`: e.g., mongodb+srv://user:pass@cluster.mongodb.net
- `MONGODB_DB`: e.g., llmbench_staging
- `MONGODB_COLLECTION_MODELS`: default models
- `MONGODB_COLLECTION_CLOUD`: e.g., metrics_cloud_staging
- `MONGODB_COLLECTION_ERRORS`: default errors_cloud
- `MONGODB_COLLECTION_BENCH_JOBS`: default bench_jobs
- `MONGODB_COLLECTION_MODEL_HEALTH`: default bench_model_health
- `MONGODB_COLLECTION_SCHEDULER_HEARTBEATS`: default bench_scheduler_heartbeats
- `MONGODB_COLLECTION_MODEL_STATUS`: default model_status

Create indexes
- With the script:
  ```bash
  MONGODB_COLLECTION_CLOUD=metrics_cloud_staging mongosh "$MONGODB_URI/$MONGODB_DB" scripts/mongo_indexes.js
  ```

Seed a model (staging/testing only - use `../manage-models.sh` for production)
- Example:
  ```bash
  PROVIDER=openai MODEL_ID=gpt-4o-mini mongosh "$MONGODB_URI/$MONGODB_DB" scripts/seed_model.js
  ```

Enqueue a manual scheduler job
- Preferred CLI:
  ```bash
  uv run env PYTHONPATH=api python -m llm_bench.scheduler.cli enqueue --provider openai --model gpt-4o-mini
  ```
- `mongosh` helper:
  ```bash
  PROVIDER=openai MODEL=gpt-4o-mini mongosh "$MONGODB_URI/$MONGODB_DB" scripts/enqueue_job.js
  ```

Lifecycle status report (dry-run by default)
- Example:
  ```bash
  MONGODB_URI=... MONGODB_DB=llm-bench scripts/model_status_report.sh --provider vertex --json
  ```
  Add `--apply --yes` to persist into the `model_status` collection once you review the output.
