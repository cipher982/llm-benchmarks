#!/bin/bash

# Array to store the response codes for each model
declare -A modelStatus

# Get top 10 trending models
models=$(curl -s -G "https://huggingface.co/api/models" \
    --data-urlencode "sort=downloads" \
    --data-urlencode "direction=-1" \
    --data-urlencode "limit=10" \
    --data-urlencode "filter=text-generation" | jq -r '.[].id')

# Run benchmarks on each model
for model in $models; do
    echo "Running benchmark for model: $model"
    responseCode=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://localhost:5000/benchmark/$model")
    echo "Finished benchmark for model: $model with HTTP Response Code: $responseCode"

    # Store the response code in the array
    modelStatus["$model"]=$responseCode
done

echo "All benchmark runs are finished."

# Print out the models and their respective response codes
echo "Summary of benchmark runs:"
for model in "${!modelStatus[@]}"; do
  echo "Model: $model, HTTP Response Code: ${modelStatus[$model]}"
done
