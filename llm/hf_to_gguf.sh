# CONVERT THE HUGGINGFACE MODELS TO GGUF FORMAT

# Unlock ultra-fast performance on your fine-tuned LLM using the Llama.cpp library on local hardware.
# Llama.cpp stands as an inference implementation of various LLM architecture models,
# implemented purely in C/C++ which results in very high performance.

# Take a clone of llama.cpp source and compile it as below
# When the build completes, main and quantize (two executables) appear in llama.cpp folder
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make CUBLAS=1 # GPU support
cd ..

# Install the required Llama.cpp dependencies
# pip3 install -r requirements.txt

# --------------------------------------------------------

# Llama.cpp expects the llm model in a 'gguf' format.
# While you can find models in this format, sometimes you might need to change your PyTorch model weights into the gguf format.
# Llama.cpp comes with a converter script to do this.
# The gguf format is recently new, published in Aug 23. It is used to load the weights and run the cpp code.
# This is a mandatory step in order to be able to later on load the model into llama.cpp.

# Download the weights of any model from HuggingFace that is based on one of the llama.cpp supported model architectures
huggingface-cli download giacomo-frisoni/positive-claim-generator --local-dir merged_models
# Check the models folder to make sure everything downloaded

# Create the tokenizer.model file
#python3 llama.cpp/convert.py merged/positive-claim-generator \
#    --vocab-only \
#    --outfile merged/positive-claim-generator/tokenizer.model \
#    --vocabtype bpe

# Run the llama.cpp convert script (make the porting).
# This script takes the original .pth files and switches them to .gguf format.
# The convert.py script is intelligent enough to find all the shards of the model
# as well as the vocab file tokenizer.model and covert them inot one GGUF model file.
python3 ./llama.cpp/convert.py merged_models \
    --outtype f16
# You should see a file named ggml-model-f16.gguf

# Quantization of deep neural networks is the process of taking full precision weights, 32bit floating points,
# and convert them to smaller approximate representation like 4bit /8 bit etc..
# Small models use less VRAM and run much faster.
# Several quantization methods are supported by llama.cpp to shrink models, resulting in different model disk size and inference speed.
# Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
# Choosing a smaller bit point makes the model run faster but might sacrifice a bit of accuracy.
# We'll use q4_1, which balances speed and accuracy well.
./llama.cpp/quantize merged_models/ggml-model-f16.gguf \
    merged_models/positive-claim-generator-7b-f16-q4_1.gguf \
    q4_1
# Each weight layer should get about 7x smaller, so the final size should be 1/7 of the original!

# Let's compare the size of the original and the quantized model
ls ./merged_models
