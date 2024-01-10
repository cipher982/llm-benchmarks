#!/bin/bash

# Define the model names
model_names=("Llama-2-7b-chat-hf" "Llama-2-13b-chat-hf" "Llama-2-70b-chat-hf")

# Activate the conda environment
activate bench

# Set the environment variable
export HUGGINGFACE_HUB_CACHE="/tmp"

# Loop over the model names
for model in "${model_names[@]}"
do
  python ./create_model.py -m meta-llama/${model}
done
