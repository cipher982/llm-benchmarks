// Creates required indexes for headless scheduler.
// Uses env var collection names if provided, with sensible defaults.

const modelsColl = process.env.MONGODB_COLLECTION_MODELS || 'models';
const metricsColl = process.env.MONGODB_COLLECTION_CLOUD || 'metrics_cloud_staging';
const errorsColl = process.env.MONGODB_COLLECTION_ERRORS || 'errors_cloud';
const jobsColl = process.env.MONGODB_COLLECTION_JOBS || 'jobs';
const statusColl = process.env.MONGODB_COLLECTION_MODEL_STATUS || 'model_status';

print(`Using DB: ${db.getName()}`);
print(`Collections: models=${modelsColl}, metrics=${metricsColl}, errors=${errorsColl}, jobs=${jobsColl}, status=${statusColl}`);

// models
// Optimized index: put enabled first since queries filter on it
// Also keeps provider/model_id for other queries
db.getCollection(modelsColl).createIndex({ enabled: 1, provider: 1, model_id: 1 });
print('Created index on models (enabled, provider, model_id)');

// metrics
db.getCollection(metricsColl).createIndex({ provider: 1, model_name: 1, gen_ts: -1 });
print('Created index on metrics (provider, model_name, gen_ts)');

// errors
db.getCollection(errorsColl).createIndex({ provider: 1, model_name: 1, ts: -1 });
print('Created index on errors (provider, model_name, ts)');

// jobs
db.getCollection(jobsColl).createIndex({ status: 1, created_at: 1 });
print('Created index on jobs (status, created_at)');

// lifecycle status
db.getCollection(statusColl).createIndex({ provider: 1, model_id: 1 }, { unique: true });
db.getCollection(statusColl).createIndex({ status: 1, computed_at: -1 });
print('Created indexes on model_status (provider/model_id unique, status/computed_at)');
