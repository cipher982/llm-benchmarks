// Cleans up staging collections (USE WITH CAUTION!)
// Env: MONGODB_COLLECTION_CLOUD, MONGODB_COLLECTION_ERRORS, MONGODB_COLLECTION_JOBS

const metricsColl = process.env.MONGODB_COLLECTION_CLOUD;
const errorsColl = process.env.MONGODB_COLLECTION_ERRORS;
const jobsColl = process.env.MONGODB_COLLECTION_JOBS;

// Safety check - only cleanup collections with "staging" in name
function isStagingCollection(name) {
  return name && (name.includes('staging') || name.includes('test') || name.includes('dev'));
}

print(`DB: ${db.getName()}`);

if (metricsColl && isStagingCollection(metricsColl)) {
  const result = db.getCollection(metricsColl).deleteMany({});
  print(`Cleaned ${metricsColl}: ${result.deletedCount} documents removed`);
} else {
  print(`Skipped metrics collection: ${metricsColl} (not a staging collection)`);
}

if (errorsColl && isStagingCollection(errorsColl)) {
  const result = db.getCollection(errorsColl).deleteMany({});
  print(`Cleaned ${errorsColl}: ${result.deletedCount} documents removed`);
} else {
  print(`Skipped errors collection: ${errorsColl} (not a staging collection)`);
}

if (jobsColl && isStagingCollection(jobsColl)) {
  const result = db.getCollection(jobsColl).deleteMany({});
  print(`Cleaned ${jobsColl}: ${result.deletedCount} documents removed`);
} else {
  print(`Skipped jobs collection: ${jobsColl} (not a staging collection)`);
}