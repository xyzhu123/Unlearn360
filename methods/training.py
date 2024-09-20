from methods.utils import max_entropy_loss, lm_loss

from tqdm import tqdm
from itertools import islice
import wandb
import json
import os
import math

def run_max_entropy(
      model,
      forget_dataloader,
      retain_dataloader,
      optimizer,
      accelerator,
      args,  
      wandb=None,
):
    model.train()
    model.zero_grad()

    local_log = {
        "total_loss": [],
        "retain_loss": [],
        "forget_loss": []
    }

    total_loss = 0.0
    total_retain_loss = 0.0
    total_forget_loss = 0.0

    total_steps = math.ceil(args.max_unlearn_steps / args.gradient_accumulation_steps)

    with tqdm(total=total_steps, desc="Training", leave=True) as pbar:
        for idx, (forget_batch, retain_batch) in enumerate(zip(forget_dataloader, retain_dataloader)):
            if idx >= args.max_unlearn_steps: 
                break

            retain_batch_squeezed = {
                key: value.squeeze() 
                for key, value in retain_batch.items() 
                if key in {"input_ids", "labels", "attention_mask"}
            }
            outputs = model(**retain_batch_squeezed, output_hidden_states=False)
            retain_loss = (
                lm_loss(outputs.logits, retain_batch_squeezed["labels"], model.config.vocab_size) / args.gradient_accumulation_steps
            )
            accelerator.backward(retain_loss)

            forget_batch_squeezed = {
                key: value.squeeze()
                for key, value in forget_batch.items()
                if key in ["input_ids", "labels", "attention_mask"]
            }
            outputs = model(**forget_batch_squeezed, output_hidden_states=False)
            forget_loss = (
                max_entropy_loss(outputs.logits) / args.gradient_accumulation_steps
            )
            accelerator.backward(forget_loss)

            total_retain_loss += retain_loss.item()
            total_forget_loss += forget_loss.item()
            total_loss += retain_loss.item() + forget_loss.item()

            if (idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()

                if wandb:
                    wandb.log({
                        "loss": total_loss, 
                        "retain_loss": total_retain_loss, 
                        "forget_loss": total_forget_loss
                    })

                local_log["total_loss"].append(total_loss)
                local_log["retain_loss"].append(total_retain_loss)
                local_log["forget_loss"].append(total_forget_loss)

                pbar.set_description(f"Total Loss: {total_loss:.4f}, Retain Loss: {total_retain_loss:.4f}, Forget Loss: {total_forget_loss:.4f}")
                pbar.update(1)

                total_loss = 0.0
                total_retain_loss = 0.0
                total_forget_loss = 0.0

    with open('log.json', "w") as f:
        json.dump(local_log, f, indent=4)
        print(f"Log saved to log.json")

    return model