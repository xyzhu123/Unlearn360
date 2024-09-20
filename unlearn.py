import torch
import numpy as np

from methods.training import run_max_entropy
from utils.dataloaders import get_text_dataloader
from utils.model_utils import load_model, save_model

import schedulefree
from accelerate import Accelerator
import random
import argparse
import wandb
import os

METHOD_CACHE = {
    "max-entropy": run_max_entropy,
}

def get_args(): 
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--wandb', type=bool, default=False)
    
    ### model args
    parser.add_argument('--model_name_or_path', type=str, default='LLM360/CrystalChat')
    parser.add_argument('--use_lora', type=bool, default=False)
    parser.add_argument('--output_path', type=str, default='./models')
    
    # data args
    parser.add_argument('--forget_topic', type=str, default='bio-forget', help='forget set')
    parser.add_argument('--retain_topic', type=str, default='wikitext-test', help='retain set')
    parser.add_argument("--min_len", type=int, default=50)
    parser.add_argument("--max_len", type=int, default=1000)

    # unlearn args
    parser.add_argument('--unlearn_method', type=str, default='max-entropy') 
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps.")                                                                                                        
    parser.add_argument("--max_unlearn_steps", type=int, default=100, help="Max number of unlearning steps.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size of unlearning.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=5e-3, help="Unlearning LR.")
    
    args = parser.parse_args()
    return args

def fix_random(random_seed=42):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

def main():

    args = get_args()
    fix_random(args.random_seed)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    model, tokenizer = load_model(args.model_name_or_path, use_lora=args.use_lora)

    forget_dataloader = get_text_dataloader(
        args.forget_topic,
        tokenizer,
        min_len=args.min_len,
        max_len=args.max_len,
        num_samples=args.max_unlearn_steps * args.batch_size,    
        batch_size=args.batch_size
    )
    
    retain_dataloader = get_text_dataloader(
        args.retain_topic,
        tokenizer,
        min_len=args.min_len,
        max_len=args.max_len,
        num_samples=args.max_unlearn_steps * args.batch_size, 
        batch_size=args.batch_size
    )
    
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=args.lr, warmup_steps=args.warmup_steps)

    model, optimizer, forget_dataloader, retain_dataloader = accelerator.prepare(model, optimizer, forget_dataloader, retain_dataloader)

    if args.unlearn_method in METHOD_CACHE:
        unlearn_method = METHOD_CACHE[args.unlearn_method]
    else:
        raise ValueError(f"Unlearn method {args.unlearn_method} not supported")
    
    run_name = name = f"{args.unlearn_method}_{args.forget_topic}_{args.retain_topic}_{args.model_name_or_path.split('/')[-1]}"

    use_wandb = None
    if args.wandb:
        if accelerator.is_main_process:
            wandb.init(project="unlearn360", name=run_name)
            use_wandb = wandb
    
    unlearned_model = unlearn_method(
        model,
        forget_dataloader,
        retain_dataloader,
        optimizer,
        accelerator,
        args,
        use_wandb
    )

    if accelerator.is_main_process:
        output_path = os.path.join(args.output_path, run_name)
        save_model(unlearned_model, tokenizer, output_path)
        print(f"Model saved to {output_path}")

if __name__ == "__main__":
    main()