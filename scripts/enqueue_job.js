// Enqueues a single ad-hoc job.
// Env: PROVIDER, MODEL (or MODEL_ID/MODEL_NAME), IGNORE_FRESHNESS (true/false),
//      MONGODB_COLLECTION_JOBS (default jobs)

const collName = process.env.MONGODB_COLLECTION_JOBS || 'jobs';
const provider = process.env.PROVIDER;
const model = process.env.MODEL || process.env.MODEL_ID || process.env.MODEL_NAME;
const ignoreFresh = (process.env.IGNORE_FRESHNESS || 'false').toLowerCase() === 'true';

if (!provider || !model) {
  throw new Error('Missing env: PROVIDER and MODEL (or MODEL_ID/MODEL_NAME) are required');
}

const job = {
  provider: provider,
  model: model,
  ignore_freshness: ignoreFresh,
  created_at: new Date(),
  status: 'pending',
};

db.getCollection(collName).insertOne(job);
print(`Enqueued job in ${collName}: ${provider}:${model} (ignore_freshness=${ignoreFresh})`);

