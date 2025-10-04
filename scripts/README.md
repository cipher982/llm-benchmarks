Scripts to help prepare MongoDB for the headless scheduler.

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
- `MONGODB_COLLECTION_JOBS`: default jobs

Create indexes
1) One-liner (example; adjust names):
   mongosh "$MONGODB_URI/$MONGODB_DB" --eval 'db.models.createIndex({provider:1, model_id:1, enabled:1});' --eval 'db.metrics_cloud_staging.createIndex({provider:1, model_name:1, gen_ts:-1});' --eval 'db.errors_cloud.createIndex({provider:1, model_name:1, ts:-1});' --eval 'db.jobs.createIndex({status:1, created_at:1});'

2) Or with the script (uses env var names):
   MONGODB_COLLECTION_CLOUD=metrics_cloud_staging mongosh "$MONGODB_URI/$MONGODB_DB" scripts/mongo_indexes.js

Seed a model (staging/testing only - use `../manage-models.sh` for production)
- Example (enable an OpenAI model):
  PROVIDER=openai MODEL_ID=gpt-4o-mini mongosh "$MONGODB_URI/$MONGODB_DB" scripts/seed_model.js

Enqueue a job
- Example (runs regardless of freshness):
  PROVIDER=openai MODEL=gpt-4o-mini IGNORE_FRESHNESS=true mongosh "$MONGODB_URI/$MONGODB_DB" scripts/enqueue_job.js

