// Creates required indexes for headless scheduler.
// Uses env var collection names if provided, with sensible defaults.

const modelsColl = process.env.MONGODB_COLLECTION_MODELS || 'models';
const metricsColl = process.env.MONGODB_COLLECTION_CLOUD || 'metrics_cloud_staging';
const errorsColl = process.env.MONGODB_COLLECTION_ERRORS || 'errors_cloud';
const jobsColl = process.env.MONGODB_COLLECTION_JOBS || 'jobs';

print(`Using DB: ${db.getName()}`);
print(`Collections: models=${modelsColl}, metrics=${metricsColl}, errors=${errorsColl}, jobs=${jobsColl}`);

// models
db.getCollection(modelsColl).createIndex({ provider: 1, model_id: 1, enabled: 1 });
print('Created index on models (provider, model_id, enabled)');

// metrics
db.getCollection(metricsColl).createIndex({ provider: 1, model_name: 1, gen_ts: -1 });
print('Created index on metrics (provider, model_name, gen_ts)');

// errors
db.getCollection(errorsColl).createIndex({ provider: 1, model_name: 1, ts: -1 });
print('Created index on errors (provider, model_name, ts)');

// jobs
db.getCollection(jobsColl).createIndex({ status: 1, created_at: 1 });
print('Created index on jobs (status, created_at)');

