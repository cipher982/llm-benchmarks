#!/bin/bash

declare -A modelStatus

models=$(curl -s -G "https://huggingface.co/api/models" \
    --data-urlencode "sort=downloads" \
    --data-urlencode "direction=-1" \
    --data-urlencode "limit=10" \
    --data-urlencode "filter=text-generation" | jq -r '.[].id')

for model in $models; do
    # Encode the forward slashes in the model name
    encodedModel=$(echo "$model" | sed 's/\//%2F/g')

    echo "Running benchmark: $model"
    responseCode=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://localhost:5000/benchmark/$encodedModel")
    echo "Finished benchmark: $model with Status Code: $responseCode"

    modelStatus["$model"]=$responseCode
done

echo "All benchmark runs are finished."

echo "Summary of benchmark runs:"
for model in "${!modelStatus[@]}"; do
  echo "Model: $model, HTTP Response Code: ${modelStatus[$model]}"
done
