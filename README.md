# llm-benchmarks
Benchmarking LLM Inference Speeds

## Summary
A collection of benchmarks for various open-source models to get an idea of the relation between different models and the inference speeds we could hope to see. There are a few things to be aware of here:

This is likely on the lower-end of what to expect. I have performed no optimizations with software or hardware, just using raw PyTorch code as loaded through the HuggingFace model hub.

Performance can vary. There is an initial response time from calling the model and smaller responses may lead to slower perceived speeds due to this overhead. With this in mind I generated responses in groups of 20 with a high temperature to get different lengths.

## Details
I used a few different methods to get complete results. First round was performed using OpenAI libraries in Python to make N calls and average those results. Next I wanted to test locally starting with my 3090. Using HuggingFace I was able to obtain model weights and load them into the Transformers library for inference. I used a high temperature to get varying lengths of outputs, and collected and tracked results using Weights and Biases. Next I rented both an A10 and an H100 from Lambda Cloud to test enterprise style GPUs. After a bit of fiddling with CUDA I was able to get some results quickly.
