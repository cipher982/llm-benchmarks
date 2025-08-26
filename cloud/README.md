# How to Run
1. Create an `.env` file from `.env.example` and fill in MongoDB and provider API keys.
2. From the repo root, run `DOCKER_BUILDKIT=1 docker compose up --build` to start the headless scheduler (no HTTP API).

## Development
Use the root compose as needed; no separate dev compose is provided.
Results are written only to MongoDB; there are no JSON file logs in the headless path.

## Logging
- Console: one concise line per model (skipped/success/error).
- Mongo: successes go to `MONGODB_COLLECTION_CLOUD`; failures go to `MONGODB_COLLECTION_ERRORS`.

## Optional Jobs
- Enqueue a job: `db.jobs.insertOne({provider:"openai", model:"gpt-4o-mini", ignore_freshness:true, created_at:new Date(), status:"pending"})`.
- Jobs are drained before periodic sampling. They run regardless of freshness if `ignore_freshness` is true.

## Indexes (mongosh)
- `db.models.createIndex({provider:1, model_id:1, enabled:1})`
- `db.<MONGODB_COLLECTION_CLOUD>.createIndex({provider:1, model_name:1, gen_ts:-1})`
- `db.errors_cloud.createIndex({provider:1, model_name:1, ts:-1})`
- `db.jobs.createIndex({status:1, created_at:1})`
