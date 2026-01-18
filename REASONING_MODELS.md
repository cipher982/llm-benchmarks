# OpenAI Reasoning Models Support

This document describes support for OpenAI's reasoning models (o1, o3, o4 series) in the benchmarking system.

## Overview

Reasoning models (o1, o3, o4 series) require the OpenAI Responses API (`/v1/responses`) instead of the Chat Completions API (`/v1/chat/completions`). The implementation automatically detects reasoning models and uses the appropriate endpoint.

## Implementation

### Model Detection

The provider automatically detects reasoning models based on their name prefix:

```python
REASONING_MODEL_PREFIXES = ("o1", "o3", "o4")

def _is_reasoning_model(model_name: str) -> bool:
    """Check if model is a reasoning model that requires Responses API."""
    return any(model_name.startswith(prefix) for prefix in REASONING_MODEL_PREFIXES)
```

Models that match these prefixes will use the Responses API.

### API Differences

| Feature | Chat Completions | Responses API |
|---------|------------------|---------------|
| Endpoint | `/v1/chat/completions` | `/v1/responses` |
| Token limit param | `max_completion_tokens` | `max_output_tokens` |
| Input format | `messages` array | `input` string |
| Response format | `choices` array | `output` array |
| Streaming events | Delta chunks | Semantic events |
| Reasoning tokens | Not exposed | Included in usage |

### Token Counting

Responses API provides `usage.output_tokens` which includes:
- **Visible output tokens**: Text generated for the user
- **Reasoning tokens**: Internal reasoning (not visible)

For benchmarking, we use `output_tokens` (total) to measure model performance, as reasoning is part of the generation process.

Example from o3-mini:
```
input_tokens: 19
output_tokens: 512
  └─ reasoning_tokens: 256  (internal)
  └─ visible tokens: 256    (shown to user)
```

### Streaming Events

The Responses API uses semantic events instead of delta chunks:

| Event Type | Description |
|------------|-------------|
| `response.created` | Response started |
| `response.in_progress` | Generation in progress |
| `response.output_text.delta` | Text chunk (with timing) |
| `response.completed` | Generation finished successfully |
| `response.incomplete` | Hit token limit |
| `response.failed` | Generation failed |

The implementation tracks timing from `response.output_text.delta` events.

### Edge Cases

1. **No visible text output**: Some reasoning models (e.g., o3-mini with low token limits) may use all tokens for reasoning and emit no visible text. This is valid - `output_tokens` will be non-zero but `time_to_first_token` will be 0.

2. **Incomplete responses**: If `max_output_tokens` is too low, the response will be marked `incomplete`. The usage metrics are still captured correctly.

3. **Temperature parameter**: Some reasoning models reject the `temperature` parameter. The implementation automatically retries without it.

## Testing

Three test scripts are provided:

### 1. `test_responses_api.py`
Tests the raw OpenAI Responses API to understand behavior:
```bash
uv run python test_responses_api.py
```

### 2. `test_openai_provider.py`
Tests the provider implementation:
```bash
uv run python test_openai_provider.py
```

### 3. `test_reasoning_models_e2e.py`
Comprehensive end-to-end test suite:
```bash
uv run python test_reasoning_models_e2e.py
```

## Adding New Models

To add a new reasoning model to the benchmark system:

1. **Add to MongoDB models collection**:
```bash
mongosh "$MONGODB_URI" --eval '
db.models.insertOne({
  provider: "openai",
  model_id: "o5-mini",  # New model
  enabled: true,
  deprecated: false,
  created_at: new Date()
})
'
```

2. **No code changes needed**: The prefix-based detection automatically handles new o1/o3/o4 models.

3. **Test the model**:
```bash
uv run python test_openai_provider.py  # Update test_cases list
```

## Troubleshooting

### Error: "only supported in v1/responses"

This indicates a model requires the Responses API but wasn't detected. Check if the model name matches one of the prefixes: `o1`, `o3`, `o4`.

### Zero tokens generated

Check if the model is using all tokens for reasoning. Increase `max_tokens` in the run config (default is 64, reasoning models may need 256+).

### Missing usage metrics

The Responses API should always include usage data. If missing, check:
1. Response object exists (`response_obj` not None)
2. Event stream completed (received `response.completed` or `response.incomplete`)
3. OpenAI API version is up to date (`openai>=1.63.2`)

## References

- [OpenAI Responses API Documentation](https://platform.openai.com/docs/api-reference/responses/create?api-mode=responses)
- [Streaming Responses Guide](https://platform.openai.com/docs/guides/streaming-responses)
- [Responses vs Chat Completions](https://platform.openai.com/docs/guides/responses-vs-chat-completions)
- [o-series Models](https://platform.openai.com/docs/models/o1)
