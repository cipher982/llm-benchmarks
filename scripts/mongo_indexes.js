// Creates required indexes for the Mongo-backed benchmark scheduler.
// Uses env var collection names if provided, with sensible defaults.

const modelsColl = process.env.MONGODB_COLLECTION_MODELS || 'models';
const metricsColl = process.env.MONGODB_COLLECTION_CLOUD || 'metrics_cloud_staging';
const errorsColl = process.env.MONGODB_COLLECTION_ERRORS || 'errors_cloud';
const jobsColl = process.env.MONGODB_COLLECTION_BENCH_JOBS || 'bench_jobs';
const healthColl = process.env.MONGODB_COLLECTION_MODEL_HEALTH || 'bench_model_health';
const heartbeatsColl = process.env.MONGODB_COLLECTION_SCHEDULER_HEARTBEATS || 'bench_scheduler_heartbeats';
const statusColl = process.env.MONGODB_COLLECTION_MODEL_STATUS || 'model_status';

print(`Using DB: ${db.getName()}`);
print(`Collections: models=${modelsColl}, metrics=${metricsColl}, errors=${errorsColl}, jobs=${jobsColl}, health=${healthColl}, heartbeats=${heartbeatsColl}, status=${statusColl}`);

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

// scheduler jobs
db.getCollection(jobsColl).createIndex({ provider: 1, status: 1, not_before: 1, priority: -1, created_at: 1 });
db.getCollection(jobsColl).createIndex({ status: 1, lease_expires_at: 1 });
db.getCollection(jobsColl).createIndex({ job_kind: 1, updated_at: -1 });
print('Created indexes on bench_jobs');

// scheduler health
db.getCollection(healthColl).createIndex({ provider: 1, model_id: 1 }, { unique: true });
db.getCollection(healthColl).createIndex({ freshness_status: 1, provider: 1 });
db.getCollection(healthColl).createIndex({ updated_at: -1 });
print('Created indexes on bench_model_health');

// scheduler heartbeats
db.getCollection(heartbeatsColl).createIndex({ component: 1 }, { unique: true });
print('Created index on bench_scheduler_heartbeats (component unique)');

// lifecycle status
db.getCollection(statusColl).createIndex({ provider: 1, model_id: 1 }, { unique: true });
db.getCollection(statusColl).createIndex({ status: 1, computed_at: -1 });
print('Created indexes on model_status (provider/model_id unique, status/computed_at)');
