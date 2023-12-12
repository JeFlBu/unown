# Main changes are highlighted with "<--"

# This script fine-tunes LLaMA-2 on the FEVER dataset (train set only, no evaluation)
# using a single GPU for positive or negative claim generation.
# WandB is used for experiment tracking and model logging.
# Trained models are pushed to the HF hub.


"""
LLAMA-2 FEATURES:
- Training Corpus: Trained on a massive 2 trillion tokens, this model is no slouch when it comes to the breadth of data it's been exposed to.
- Context Length: With an increased context length of 4K, the model can grasp and generate extensive content.
- Grouped Query Attention (GQA): Adopted to improve inference scalability, GQA accelerates attention computation by caching previous token pairs.
- Performance: This model shines in various benchmarks, standing tall against competitors like Llama 1 65B and even Falcon models.
"""


"""
---------------------------------------------------
INSTALLATIONS AND IMPORTS SECTION

This section of the code is dedicated to importing necessary Python libraries and modules for the execution of the fine-tuning script.
Please ensure all the libraries are installed correctly before running the code.
---------------------------------------------------
"""

import sys
import os

# We assume bitsandbytes being cloned from the GitHub repository and compiled with GPU support
# Not being installed from PyPI distributions, we need to specify the path where the package is located
sys.path.append("/workspace/bitsandbytes/")

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

import transformers

from trl import SFTTrainer

from datasets import load_from_disk

import bitsandbytes as bnb

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser
)

import wandb

from huggingface_hub import login


"""
---------------------------------------------------
GENERAL UTILS SECTION
---------------------------------------------------
"""

def print_header_message(message):
    print("=" * 60)
    print(message)
    print("=" * 60, "\n\n")

def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"all params: {all_params:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_params:.2f}")


"""
---------------------------------------------------
COMMAND-LINE ARGUMENTS SECTION

This section of the code is dedicated to defining and parsing command-line arguments.
These arguments can be specified when running the script from the bash command line.

Usage:
python3 llama2_feverous_finetuning.py --run_name negative_claim_generator_feverous_adapter --data_type negative --model_name giacomo-frisoni/negative-claim-generator

Inspired by: https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da
---------------------------------------------------
"""

@dataclass
class ScriptArguments:
    """
    Claim generator hyperparameters.
    These arguments vary depending the GPU capacity and what their features are, and what size model you want to train.
    """

    # --- mandatory ---

    run_name: str = field(metadata={"help": "The name with which you want to log your run and save your model on the HF hub"})
    data_type: str = field(metadata={"help": "The type of training data to use; choose either 'positive' or 'negative'"})

    # --- optional ---

    # logging
    hf_token: Optional[str] = field(default="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", metadata={"help": "Your HF User Access Token; you can generate one at https://huggingface.co/settings/tokens"})
    wandb_token: Optional[str] = field(default="e9d646d5f5f79597160b9afa9fd04ee6c8e85ac8", metadata={"help": "Your W&B User Access Token; you can get it from https://wandb.ai/authorize"})
    wandb_project: Optional[str] = field(default="unown", metadata={"help": "The 'wandb' project in which logging the run"})
    model_name: Optional[str] = field(default="giacomo-frisoni/positive-claim-generator", metadata={"help": "The model you want to finetune"}) # <--
    logging_strategy: Optional[str] = field(default="steps", metadata={"help": "The logging strategy to adopt during training; 'steps': logging is done every 'logging_steps'"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "How often to log to W&B during training"})

    # model loading
    use_4bit: Optional[bool] = field(default=True, metadata={"help": "Activate 4bit precision base model loading"})
    use_nested_quant: Optional[bool] = field(default=True, metadata={"help": "Activate nested quantization for 4bit base models"})
    bnb_4bit_compute_dtype: Optional[str] = field(default="bfloat16", metadata={"help": "Compute dtype for 4bit base models"})
    bnb_4bit_quant_type: Optional[str] = field(default="nf4", metadata={"help": "Quantization type fp4 or nf4"})

    # training (general)
    resume_checkpoint: Optional[str] = field(default=None, metadata={"help": "If training was previously interrupted, this can be set to the path of a saved checkpoint"})
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={"help": "When set to None, it implies that there is no preset limit, and it might use the default limit of the model or process sequences of any length present in the dataset."}
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "A boolean flag that indicates whether shorter sequences should be packed together to form an input of maximum sequence length. This is done to efficiently utilize memory and speed up the training process."}
    )
    batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=2)
    group_by_length: bool = field(default=True, metadata={"help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."})
    num_train_epochs: Optional[int] = field(default=3)
    max_steps: Optional[int] = field(default=-1, metadata={"help": "How many optimizer update steps to take. If set, overrides num_train_epochs."})
    max_train_samples: Optional[int] = field(default=100)
    learning_rate: Optional[float] = field(default=2e-4)
    # ---------------------------------------------------
    # working with half-precision floating-point numbers can provide several benefits, such as reducing memory consumption and speeding up model training
    # if supported by your hardware, BF16---proposed by Google---has better stability than F16 for mixed precision training
    fp16: Optional[bool] = field(default=False, metadata={"help": "Enable fp16 training"})
    bf16: Optional[bool] = field(default=True, metadata={"help": "Enable bf16 training, a format primarily used in TPUs"})
    # ---------------------------------------------------
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "Learning rate schedule. Constant is a bit better than cosine, and has advantage for analysis."},
    )
    warmup_ratio: Optional[float] = field(default=0.03, metadata={"help": "The ratio of warmup steps to total training steps"})
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    save_strategy: Optional[str] = field(default="epoch", metadata={"help": "The checkpoint save strategy to use."})
    output_dir: Optional[str] = field(default="./runs", metadata={"help": "The output directory where the model predictions and checkpoints will be written; i.e., output_dir/run_name"})

    # training (peft)
    # By tweaking these parameters, especially lora_alpha and r, you can observe how the model’s performance and resource consumption change, helping you find the optimal setup for your specific task
    lora_r: Optional[int] = field(
        default=8,
        metadata={"help": "The rank parameter of the LoRA adapter. It is essentially a measure of how the original weight matrices are broken down into simpler, smaller matrices. This reduces computational requirements and memory consumption. Lower ranks make the model faster but might sacrifice performance. The original LoRA paper suggests starting with a rank of 8, but for QLoRA, a rank of 64 is required."}
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": "The alpha parameter of the LoRA adapter. It controls the scaling of the low-rank approximation. It's like a balancing act between the original model and the low-rank approximation. Higher values might make the approximation more influential in the fine-tuning process, affecting both performance and computational cost."}
    )
    lora_dropout: Optional[int] = field(default=0.1, metadata={"help": "The dropout probability parameter of the LoRA adapter. This is the probability that each neuron's output is set to zero during training, used to prevent overfitting."})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


"""
---------------------------------------------------
CLOUD LOGIN AND TRACKING SECTION

This section of the code is dedicated to logging into cloud services and set W&B experiment tracking preferences.
Please ensure all the credentials are set correctly before running the code.
---------------------------------------------------
"""

# HF login
login(token=script_args.hf_token)

# W&B login and settings
# - Specify your Team Access Token
wandb.login(key=script_args.wandb_token)
# - Set the W&B project where this run will be logged
os.environ["WANDB_PROJECT"] = script_args.wandb_project
# - Log all model checkpoints to W&B Artifacts as 'model-{run_name}'
#   - "checkpoint": a checkpoint will be uploaded every time the model is saved
#   - "end": the model will be uploaded at the end of the training
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
# - Turn off the automatic watching of model gradients for each training step to log faster
os.environ["WANDB_WATCH"] = "false"


"""
---------------------------------------------------
DATASET LOADING SECTION

This section of the code is dedicated to loading the necessary training dataset.
In this use case, we do not have an eval set.
In fact, there is no 'save_total_limit' argument based on 'metric_for_best_model'.
---------------------------------------------------
"""

print_header_message("Loading dataset...")

if script_args.data_type == "positive":
    train_data = load_from_disk("./datasets/feverous/feverous_pos")["train"]
elif script_args.data_type == "negative":
    train_data = load_from_disk("./datasets/feverous/feverous_neg")["train"]
# Dictionary object
# "feverous_pos": ("evidences_txt", "claim", "title")
# "feverous_neg": ("evidences_txt", "claim", "title")



"""
---------------------------------------------------
MODEL LOADING

This section of the code is dedicated to creating and preparing the model.
---------------------------------------------------
"""

def load_model(args):
    # To reduce the model size and increase inference speed, we use quantization

    # BitsAndBytes from HuggingFace lets us to dynamically adjust the precision used when
    # loading a model into memory, irrespective of the precision utilized during training
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )

    if compute_dtype == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("Your GPU actually supports bfloat16, you can accelerate training with the argument --bf16")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",  # dispatch efficiently the model on the available resources
        trust_remote_code=True,
        use_auth_token=True,
    )
    # For multi-gpu training, you can set a a max VRAM per GPU as in https://blog.ovhcloud.com/fine-tuning-llama-2-models-using-a-single-gpu-qlora-and-ai-notebooks/

    # Check: https://github.com/huggingface/transformers/pull/24906
    # Check: https://artificialcorner.com/mastering-llama-2-a-comprehensive-guide-to-fine-tuning-in-google-colab-bedfcc692b7f
    model.config.pretraining_tp = 1

    # Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # Prepare the model for fine-tuning
    model = prepare_model_for_kbit_training(model)

    # PEFT is about adapting pre-trained models to new tasks with the least changes possible to the model's parameters
    # - Reduced Overfitting: Limited datasets can spell trouble. Modifying too many parameters might cause the model to overfit.
    # - Swift Training: Fewer parameter tweaks mean fewer calculations, which translates to speedier training sessions.
    # - Resource Heavy: Deep neural training is a resource intensive. PEFT minimizes the computational and memory burden, making deployments in resource-tight scenarios more practical.
    # - Preserving Knowledge: Extensive pre-training on broad datasets packs models with invaluable general knowledge. With PEFT, we ensure that this treasure trove isn’t lost when adapting the model to novel tasks.
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        r=args.lora_r,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",  # it would be "SEQ_TO_SEQ_LM" for FLAN-T5
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    # Padding tokens are special tokens added when we have a mini-batch of text seq with uneven lengths.
    # Most LLM tokenizers have a default padding-side and left padding logically makes sense as decoder-only models learn to predict next token.
    # Example with max_seq_len=6
    # - Sent 1: [[PAD], [PAD], [PAD], The, weather, is, .]
    # - Sent 2: [[PAD], [PAD], [PAD], [PAD], It's, sunny, .]
    # But there are exceptions to it, including the LLaMa family.
    # Consensus on the exchanges from repos and HF discussion forum is: default padding side is “Right” for LLAMA family models.
    # --------------------------------------------
    # LLMs are usually pre-trained without padding.
    # Nonetheless, for fine-tuning LLMs on custom datasets, padding is necessary.
    # Failing to correctly pad training examples may result in different kinds of unexpected behaviors:
    # - null loss or infinity loss during training
    # - over-generation
    # - empty output during inference
    # - etc.
    # To avoid these issues, we add padding support by defining a custom PAD token.
    # Consider this:
    # (1) If the EOS token is used for padding, during training, the model learns that
    #     the EOS token is just filler and shouldnâ€™t generate it as part of its output. This means that
    #     given a prompt, the model might generate never-ending text, and we would need to truncate the
    #     output manually.
    # (2) While there are various ways to set padding tokens in HuggingFace, some methods can introduce
    #     errors, especially at the GPU level. For instance, using the 'add_special_tokens' method might
    #     cause CUDA-related errors. This could be due to changes in embedding dimensions. But without
    #     diving deep, it's hard to pinpoint the exact cause. As a result, directly setting the
    #     'pad_token' attribute is a safer bet.
    # See: https://towardsdatascience.com/padding-large-language-models-examples-with-llama-2-199fb10df8ff
    # See: https://artificialcorner.com/mastering-llama-2-a-comprehensive-guide-to-fine-tuning-in-google-colab-bedfcc692b7f
    tokenizer.pad_token = "<PAD>"
    # Padding can be added either to the left (before the BOS) or to the right (after the EOS).
    tokenizer.padding_side = "right" # fix weird overflow issue with fp16 training

    return model, peft_config, tokenizer


print_header_message("Loading LLaMA-2...\n")

model, peft_config, tokenizer = load_model(script_args)
model.config.use_cache = False # re-enable for inference to speed up predictions for similar inputs

print_trainable_parameters(model)

"""
---------------------------------------------------
PROMPT BUILDING AND TRAINING DATA SECTION

This section of the code is dedicated to defining the prompt templates and the training data for the model.
Note that the following instruction-aware format is needed for LLaMA-chat models only.
'<s>[INST] {user_message} [/INST] {response}'
---------------------------------------------------
"""

def positive_template(sample):
    sample["text"] = f"""
Write a claim that uses the following evidence.

Evidence:
<title> {sample["title"]} <evidence> {".".join(sample["evidences_txt"]).strip()}

Claim:
{sample["claim"]}
""".strip()
    return sample

def negative_template(sample):
    sample["text"] = f"""
Write a negative claim that is false with regards to the following evidence.

Evidence:
<title> {sample["title"]} <evidence> {".".join(sample["evidences_txt"]).strip()}

Claim:
{sample["claim"]}
""".strip()
    return sample

if script_args.max_train_samples > 0:
    train_data = train_data.select(range(script_args.max_train_samples))

if script_args.data_type == "positive":
    train_data = train_data.map(positive_template)#, remove_columns=[f for f in train_data.features if not f == 'text'])
elif script_args.data_type == "negative":
    train_data = train_data.map(negative_template)#, remove_columns=[f for f in train_data.features if not f == 'text'])


"""
---------------------------------------------------
MODEL TRAINING SECTION

This section of the code is dedicated to training the model with SFT over the defined dataset.
---------------------------------------------------
"""

run_output_dir = os.path.join(script_args.output_dir, script_args.run_name)

# Training arguments
# See: https://huggingface.co/docs/transformers/main_classes/trainer
# SFTTrainer always pads by default the sequences to the max_seq_length argument of the SFTTrainer.
# If none is passed, the trainer will retrieve that value from the tokenizer.
# Some tokenizers do not provide default value, so there is a check to retrieve the minimum between 2048 and that value.
training_args = transformers.TrainingArguments(
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    optim=script_args.optim,
    learning_rate=script_args.learning_rate,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    warmup_ratio=script_args.warmup_ratio,
    max_grad_norm=script_args.max_grad_norm,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    # ---
    save_strategy=script_args.save_strategy,
    report_to="wandb",
    run_name=script_args.run_name,
    logging_strategy=script_args.logging_strategy,
    logging_steps=script_args.logging_steps,
    output_dir=run_output_dir,
)

# We use the Supervised Fine-tuning Trainer class from TRL
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=script_args.max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
    packing=script_args.packing,
)

print_header_message("Training model...\n")

# Set the logging level to "info" to see important messages during training
transformers.logging.set_verbosity_info()

trainer.train(script_args.resume_checkpoint)
trainer.model.save_pretrained(run_output_dir)

wandb.finish()

# Empty VRAM
del model
del trainer
import gc
gc.collect()
gc.collect()

"""
model
───llama-peft
│      adapter_config.json
│      adapter_model.bin
│      trainer_state.json
│
└──llama_7b
        config.json
        generation_config.json
        pytorch_model-00001-of-00002.bin
        pytorch_model-00002-of-00002.bin
        pytorch_model.bin.index.json
        special_tokens_map.json
        tokenizer.json
        tokenizer.model
        tokenizer_config.json

With this script, you have trained the llama-peft LORA adapter.
"""