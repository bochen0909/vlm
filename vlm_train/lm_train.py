from lm_dataloader import get_dataloader
import torch
import torch.nn as nn
import os
import torch.optim as optim
from tqdm import tqdm
from lm_to_vlm import LM_2_VLM
import numpy as np
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
if __name__ == "__main__":
    model_id = "vlm_peft"
    model_name = "HuggingFaceTB/SmolLM-135M-Instruct"

    train_loader, test_loader = get_dataloader(batch_size=8, tokenizer_name=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    pad_token_id = tokenizer.pad_token_id
    model = LM_2_VLM(
        model_name=model_name,
        qformer_model_path=f"models/trained_qformer_1/best",
        pad_token_id=pad_token_id)     
    model.to(device)

    # --- Optimizer Setup ---
    lr_slow = 5e-5
    lr_fast = 2e-4

    qformer_params = model.qformer.get_grouped_params()
    optimizer = optim.AdamW([
        {"params": qformer_params["default"], "lr": lr_slow},
        {"params": qformer_params["cross_blocks"], "lr": lr_slow},
        {"params": qformer_params["query_embeddings"], "lr": lr_slow},
        {"params": model.adapter.parameters(), "lr": lr_fast},
        {"params": filter(lambda p: p.requires_grad, model.llm.parameters()), "lr": lr_fast},
    ])

    # --- Training Loop ---
    epochs = 5
    log_every = 20
    save_every = 100
    step = 0
    best_test_loss = float('inf')

    def run_inference(model, test_loader, limit_batches=20):
        model.eval()
        losses = []
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                if i >= limit_batches:
                    break
                
                img = data["image"]
                prefix = data["prefix"]
                assistant = data["assistant_prompt"]
                
                output = model(img, prefix, assistant)
                losses.append(output.loss.item())
                
        model.train()
        
        if not losses:
            return float('inf')
        return np.mean(losses)

    model.train() # Set to train mode (affects dropout/batchnorm in QFormer/Adapter)

    print("Starting training...")
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for data in pbar:
            step += 1
            img = data["image"]
            prefix = data["prefix"]
            assistant = data["assistant_prompt"]
            
            optimizer.zero_grad()
            output = model(img, prefix, assistant)
            loss = output.loss
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if step % log_every == 0:
                test_loss = run_inference(model, test_loader)
                tqdm.write(f"Step {step} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss:.4f}")

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    model.save_checkpoint(f"models/{model_id}/best")
                    tqdm.write(f"New best model saved! Loss: {best_test_loss:.4f}")

        if step % save_every == 0:
            model.save_checkpoint(f"models/{model_id}/latest")

    # Save final model
    model.save_checkpoint(f"models/{model_id}/final")
    print("Training complete.")

