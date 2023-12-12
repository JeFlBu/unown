# Start from the NVIDIA CUDA image to get both CUDA and the GPU libraries
# "devel" (instead of "base") is required to get access to the CUDA development tools (nvcc, cuDNN header files, etc.)
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
LABEL maintainer="disi-unibo-nlp"

# Zero interaction (default answers to all questions)
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /workspace

# Install general-purpose dependencies
RUN apt-get update -y && \
    apt-get install -y curl \
                       git \
                       bash \
                       nano \
                       wget \
                       python3.8 \
                       python3-pip && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install wrapt --upgrade --ignore-installed
RUN pip install gdown

# Install PyTorch and related packages (part 1)
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install other Python packages
RUN pip3 install --upgrade transformers==4.30.0
RUN pip3 install --upgrade trl==0.7.2
RUN pip3 install --upgrade peft==0.5.0
RUN pip3 install --upgrade accelerate==0.23.0
RUN pip3 install --upgrade datasets==2.14.5
RUN pip3 install --upgrade wandb==0.15.7
RUN pip3 install --upgrade tokenizers==0.13.3
RUN pip3 install --upgrade tqdm==4.63.1
RUN pip3 install --upgrade nltk==3.7
RUN pip3 install --upgrade scipy==1.10.1
RUN pip3 install --upgrade huggingface_hub==0.18.0

# Install PyTorch and related packages (part 2)
RUN pip install --force-reinstall torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/ # Fix torch errors

# Back to default frontend
ENV DEBIAN_FRONTEND=dialog
