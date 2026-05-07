# How to Run
1. Configure the root `.env` file with MongoDB connection and provider API keys.
2. From the repo root, run `DOCKER_BUILDKIT=1 docker compose up --build` to start the Mongo-backed scheduler (no HTTP API).

## Adding Models
To add new cloud provider models, use `../manage-models.sh` from the parent directory. Models are loaded dynamically from MongoDB.

## Development
Use the root compose as needed; no separate dev compose is provided.
Results are written only to MongoDB; there are no JSON file logs in the scheduler path.

## Logging
- Console: scheduler, reaper, and worker status lines.
- Mongo: jobs go to `bench_jobs`, health goes to `bench_model_health`, successes go to `MONGODB_COLLECTION_CLOUD`, and failures go to `MONGODB_COLLECTION_ERRORS`.

## Manual Jobs
- Preferred: `uv run env PYTHONPATH=api python -m llm_bench.scheduler.cli enqueue --provider openai --model gpt-4o-mini`
- Mongo helper: `PROVIDER=openai MODEL=gpt-4o-mini mongosh "$MONGODB_URI/$MONGODB_DB" scripts/enqueue_job.js`

## Indexes (mongosh)
- `db.models.createIndex({provider:1, model_id:1, enabled:1})`
- `db.<MONGODB_COLLECTION_CLOUD>.createIndex({provider:1, model_name:1, gen_ts:-1})`
- `db.errors_cloud.createIndex({provider:1, model_name:1, ts:-1})`
- `db.bench_jobs.createIndex({provider:1, status:1, not_before:1, priority:-1, created_at:1})`
- `db.bench_model_health.createIndex({provider:1, model_id:1}, {unique:true})`
