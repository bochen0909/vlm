# VLLMs

- [x] Set up Huggingface, download the SLMs (Qwen3-0.6B and SmolLM-270M)
- [x] Get a small-ass vision transformer model (Vit-base)
- [x] Pick a dataset (start with conceputal-captions) 
- [ ] Write a simple *Q-Former* layer to compress image tokens  (use distillbert for text init?)
- [ ] Pretrain the Q-Former on contrastive loss (maybe generative loss too?)
- [ ] Write a linear/small adapter layer to transfer image space to llm space
- [ ] Train that shit
- [ ] Evaluate and think if further end-to-end (or LLM-only) finetuning is necessary
- [ ] Finetune Q-former, linear-layer, maybe with a *LORA* adapter? (Extra)

