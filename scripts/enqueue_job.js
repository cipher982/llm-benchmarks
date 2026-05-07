// Enqueues a single manual benchmark job in the scheduler queue.
// Env: PROVIDER, MODEL (or MODEL_ID/MODEL_NAME),
//      MONGODB_COLLECTION_BENCH_JOBS (default bench_jobs)

const collName = process.env.MONGODB_COLLECTION_BENCH_JOBS || 'bench_jobs';
const provider = process.env.PROVIDER;
const model = process.env.MODEL || process.env.MODEL_ID || process.env.MODEL_NAME;
const now = new Date();
const priority = Number(process.env.PRIORITY || 10000);
const deadlineSeconds = Number(process.env.DEADLINE_SECONDS || 120);
const maxAttempts = Number(process.env.MAX_ATTEMPTS || 2);

if (!provider || !model) {
  throw new Error('Missing env: PROVIDER and MODEL (or MODEL_ID/MODEL_NAME) are required');
}

const job = {
  _id: `manual:${provider}:${model}:${now.toISOString().replace(/[-:.]/g, '')}`,
  provider: provider,
  model_id: model,
  status: 'queued',
  priority: priority,
  attempt: 0,
  max_attempts: maxAttempts,
  deadline_seconds: deadlineSeconds,
  not_before: now,
  created_at: now,
  updated_at: now,
  started_at: null,
  lease_expires_at: null,
  worker_id: null,
  last_attempt_error_kind: null,
  last_attempt_error_message: null,
  job_kind: 'manual',
};

db.getCollection(collName).insertOne(job);
print(`Enqueued job in ${collName}: ${job._id}`);
