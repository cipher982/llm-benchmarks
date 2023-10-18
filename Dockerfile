FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install any python packages you need
COPY ./requirements.txt requirements.txt

RUN python3 -m pip install -r requirements.txt

# Install PyTorch and torchvision
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu118

# Set the working directory
WORKDIR /app

# Set the entrypoint
ENTRYPOINT [ "python3", "-m", "llm_benchmarks.api" ]
