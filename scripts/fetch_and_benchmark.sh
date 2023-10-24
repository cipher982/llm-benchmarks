#!/bin/bash

# Fetch models using curl and jq
models=$(curl -s -G "https://huggingface.co/api/models" \
    --data-urlencode "sort=downloads" \
    --data-urlencode "direction=-1" \
    --data-urlencode "limit=100" \
    --data-urlencode "filter=text-generation" | jq -r '.[].id')

# Explicitly split by newline into an array
modelArray=($(echo "$models" | tr '\n' ' '))

# Initialize an empty array to store valid models
validModels=()

# Initialize an empty array to store dropped models
droppedModels=()

# Loop to filter out the models
for model in "${modelArray[@]}"; do
  echo "Debug: Processing model -> $model"  # Debug print

  # Extract parameter count substring
  paramCount=$(echo "$model" | sed -nE 's/.*-([0-9]+[MmBb]).*/\1/p')
  echo "Debug: Extracted param count -> $paramCount"  # Debug print

  # If parameter count is empty, include it
  if [ -z "$paramCount" ]; then
    validModels+=("$model")
    continue
  fi

  # If it's in billions, check the numerical part
  if [[ ${paramCount: -1} == 'B' || ${paramCount: -1} == 'b' ]]; then
    numericalPart=${paramCount:0:${#paramCount}-1}
    if (( numericalPart > 5 )); then
      droppedModels+=("$model")
      continue
    fi
  fi

  # If it passed all checks, add it to the valid models list
  validModels+=("$model")
done

# Convert the valid model array back to a string, using space as a separator
IFS=' '
filteredModels="${validModels[*]}"
droppedModelsString="${droppedModels[*]}"

echo "Filtered models (Kept): $filteredModels"
echo "Filtered models (Dropped): $droppedModelsString"


for model in $filteredModels; do
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
