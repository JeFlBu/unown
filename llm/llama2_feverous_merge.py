"""
---------------------------------------------------
GENERAL UTILS SECTION
---------------------------------------------------
"""

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def download_wandb_adapter(wandb_api, artifact_name):
    adapter_ckpt = wandb_api.artifact(artifact_name)
    adapter_ckpt_path = adapter_ckpt.download()
    return adapter_ckpt_path

def download_run_config(wandb_api, run_name):
    run = wandb_api.run(run_name)
    config_path = run.file("config.yaml").download(replace=True)
    return config_path

def merge(adapter_path, base_model_name, merged_model_name, save_merged_local=None, save_merged_hf_hub=True):
    """
    Merges a pretrained LLM with an adapter and saves the merged model and tokenizer locally or on the Hugging Face Hub.

    Parameters:
    adapter_path (str): The path to the adapter to be merged with the model.
    base_model_name (str): The name of the base model.
    merged_model_name (str): The name of the merged model.
    save_merged_local (str, optional): The local path where the merged model will be saved. If None, the model will not be saved locally.
    save_merged_hf_hub (bool, optional): If True, the merged model will be saved on the Hugging Face Hub. Defaults to True.

    Returns:
    model (AutoPeftModelForCausalLM): The merged model.
    tokenizer (AutoTokenizer): The tokenizer.
    """
    # (Re-)load the model in FP16
    # If you have saved your adapter locally or on the Hub, you can leverage the AutoPeftModelForxxx classes
    # and load any PEFT model with a single line of code
    # See: https://huggingface.co/docs/peft/quicktour
    #model = AutoPeftModelForCausalLM.from_pretrained(
    #    adapter_path,
    #    torch_dtype=torch.bfloat16,
    #    device_map="auto")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    # (Re-)load the tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = "<PAD>"
    tokenizer.padding_side = "right"
    # Save the merged model locally
    if save_merged_local:
        os.makedirs(save_merged_local, exist_ok=True)
        model.save_pretrained(save_merged_local, safe_serialization=True)
        tokenizer.save_pretrained(save_merged_local)
    # Save the merged model on the Hub
    if save_merged_hf_hub:
        model.push_to_hub(f"giacomo-frisoni/{merged_model_name}", private=True, max_shard_size='2GB')
        tokenizer.push_to_hub(f"giacomo-frisoni/{merged_model_name}", private=True)
    return model, tokenizer


"""
---------------------------------------------------
DOWNLOAD ADAPTER CHECKPOINTS
---------------------------------------------------
"""

import os, wandb

project_name = "unown"
wandb_api_key = "e9d646d5f5f79597160b9afa9fd04ee6c8e85ac8"
pos_artifact_name = "disi-unibo-nlp-team/unown/checkpoint-positive_claim_generator_feverous_adapter:v2"
neg_artifact_name = "disi-unibo-nlp-team/unown/checkpoint-negative_claim_generator_feverous_adapter:v5"

# Login on W&B
os.environ['WANDB_API_KEY'] = wandb_api_key
wandb.login()

# Create an instance of the W&B API
api = wandb.Api()

pos_adapter_path = download_wandb_adapter(api, pos_artifact_name)
neg_adapter_path = download_wandb_adapter(api, neg_artifact_name)


"""
---------------------------------------------------
MERGE ADAPTER WEIGHTS TO BASE MODEL AND UPLOAD TO HUGGINGFACE

Once we have our fine-tuned adapter weights, we can combine them with the base weights to build our fine-tuned model.
---------------------------------------------------
"""

from huggingface_hub import login

login(token="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")

pos_base_model_name = "giacomo-frisoni/positive-claim-generator"
neg_base_model_name = "giacomo-frisoni/negative-claim-generator"

print(f"Merging {neg_base_model_name} and {neg_adapter_path}")

model, tokenizer = merge(
    neg_adapter_path,
    neg_base_model_name,
    "negative-claim-generator-feverous",
    save_merged_local=None, # merged/positive-claim-generator
    save_merged_hf_hub=True)

print(f"Merging {pos_base_model_name} and {pos_adapter_path}")

model, tokenizer = merge(
    pos_adapter_path,
    pos_base_model_name,
    "positive-claim-generator-feverous",
    save_merged_local=None, # merged/positive-claim-generator
    save_merged_hf_hub=True)