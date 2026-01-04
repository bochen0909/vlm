import torch
import os
import random
from lm_to_vlm import LM_2_VLM, device
from lm_dataloader import LMDataset, get_train_dataset
from transformers import AutoTokenizer
import subprocess
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    print(f"Using device: {device}")
    # 1. Setup Model and Tokenizer
    model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    model = LM_2_VLM(model_name = model_name, 
                     pad_token_id=tokenizer.pad_token_id)
    dataset = get_train_dataset(tokenizer_name=model_name) 
    # Check for checkpoint
    ckpt_path = "models/vlm_checkpoints/best"
    model.load_checkpoint(ckpt_path)
    model.to(device)
    model.eval()
    
    # Pick a random sample
    idx = 13 # random.randint(0, 50)
    
    sample = dataset[idx]
    image_filename = sample['image_filename']
    caption = sample['caption']
    # Prepare inputs
    img_tensor = sample["image"].unsqueeze(0).to(device) # [1, patches, dim] 
    
    # Create a fresh prompt for inference
    prompt_text = "Describe this picture: "
    print(f"\nUser Prompt: {prompt_text}")
    
    # Construct Prefix just like training
    prefix_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Answer the user's question truthfully"},
            {"role": "user", "content": prompt_text},
        ],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    assistant = sample["assistant_prompt"]
    output = model(img_tensor, prefix_ids, assistant)
    loss = output.loss
    print(idx, " loss: ", loss)


    
    # 3. Generate
    print("Generating response...")
    output_ids = model.generate(
        img=img_tensor,
        prefix_ids=prefix_ids,
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
        # repetition_penalty=1.2
    )
    # 4. Decode
    # The output_ids usually contains the newly generated tokens
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    subprocess.run(["chafa", image_filename])
    print(f"\nGenerated Response:\n{generated_text}")

if __name__ == "__main__":
    main()
