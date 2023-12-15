#!/bin/bash

# Load environment variables from .env file
env $(grep -v '^#' /home/drose/git/llm-benchmarks/.env | xargs) \
/home/drose/miniconda3/envs/bench_hf/bin/python \
/home/drose/git/llm-benchmarks/huggingface/llm_bench_huggingface/server.py \
--port 5001
