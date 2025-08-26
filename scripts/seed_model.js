// Seeds a single enabled model into the models collection.
// Env: PROVIDER, MODEL_ID, ENABLED (default true), MONGODB_COLLECTION_MODELS (default models)

const collName = process.env.MONGODB_COLLECTION_MODELS || 'models';
const provider = process.env.PROVIDER;
const modelId = process.env.MODEL_ID;
const enabled = (process.env.ENABLED || 'true').toLowerCase() !== 'false';

if (!provider || !modelId) {
  throw new Error('Missing env: PROVIDER and MODEL_ID are required');
}

const doc = {
  provider: provider,
  model_id: modelId,
  enabled: enabled,
  added_at: new Date(),
};

db.getCollection(collName).insertOne(doc);
print(`Inserted model into ${collName}: ${provider}:${modelId} (enabled=${enabled})`);

