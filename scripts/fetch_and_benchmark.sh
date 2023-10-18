#!/bin/bash

# Get top 10 trending models
models=$(curl -G "https://huggingface.co/api/models" \
    --data-urlencode "sort=downloads" \
    --data-urlencode "direction=-1" \
    --data-urlencode "limit=10" \
    --data-urlencode "filter=text-generation" | jq -r '.[].id')

# Run benchmarks on each model
for model in $models; do
    curl -X POST "http://localhost:5000/benchmark/$model"
done
