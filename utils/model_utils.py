import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig

# load & save model

def load_model(model_name_or_path, use_lora=False, lora_rank=32):

    # load model
    model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    )
    if use_lora:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            target_modules=["c_attn"],
            r=lora_rank,
            lora_alpha=16,
        )
        model.add_adapter(peft_config)
        model.enable_adapters()
        
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    return model, tokenizer

def save_model(model, tokenizer, path):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

