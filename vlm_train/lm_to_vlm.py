from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from q_former import QFormer
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

class LM_2_VLM(nn.Module):
    def __init__(self, model_name, 
                 qformer_model_path="models/trained_qformer_1/best",
                 pad_token_id=None):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16).to(device)
        
        self.llm.eval()
        for params in self.llm.parameters():
            params.requires_grad = False

        self.qformer = QFormer.from_pretrained(qformer_model_path).to(device)
        self.adapter = nn.Linear(in_features=self.qformer.hidden_size, 
                                 out_features=self.llm.config.hidden_size).to(device)
        self.pad_token_id = pad_token_id or self.llm.config.eos_token_id

    def forward(self, img, prefix_ids, assistant_ids):
        img_emb, _ = self.qformer.encode_image(img)
        img_emb = self.adapter(img_emb)
        img_emb = img_emb.to(dtype=self.llm.dtype)

        prefix_emb = self.llm.get_input_embeddings()(prefix_ids)
        assistant_emb = self.llm.get_input_embeddings()(assistant_ids)
        
        input_embs = torch.cat([prefix_emb, 
                       img_emb, 
                       assistant_emb],
                     dim=1)

        attention_mask = torch.cat([
            (prefix_ids!=self.pad_token_id).long(),
            torch.ones(img_emb.size(0), img_emb.size(1), device=device).long(),
            (assistant_ids!=self.pad_token_id).long()
        ], dim=1)

        # Calculate position_ids to handle left-padding (ignore padding tokens in count)
        position_ids = attention_mask.cumsum(dim=1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0) # Clamp padding positions to 0
        
        # Construct Labels
        # -100 means loss is ignored for these tokens
        prefix_labels = torch.full_like(prefix_ids, -100)
        image_labels = torch.full((img_emb.shape[0], img_emb.shape[1]), -100, device=device, dtype=torch.long)
        assistant_labels = assistant_ids.clone()
        assistant_labels[assistant_ids == self.pad_token_id] = -100
        
        labels = torch.cat([prefix_labels, image_labels, assistant_labels], dim=1)
        
        output = self.llm(inputs_embeds=input_embs,
                          attention_mask=attention_mask,
                          position_ids=position_ids,
                          labels=labels)
        
        return output

    def save_checkpoint(self, path):
        os.makedirs(path, exist_ok=True)
        # Save QFormer using its native method
        self.qformer.save_pretrained(os.path.join(path, "qformer"))
        # Save Adapter
        torch.save(self.adapter.state_dict(), os.path.join(path, "adapter.pt"))
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        # Load QFormer weights
        qformer_path = os.path.join(path, "qformer", "pytorch_model.bin")
        if os.path.exists(qformer_path):
            state_dict = torch.load(qformer_path, map_location=device)
            self.qformer.load_state_dict(state_dict)
            print("Loaded QFormer weights.")
        
        # Load Adapter weights
        adapter_path = os.path.join(path, "adapter.pt")
        if os.path.exists(adapter_path):
            self.adapter.load_state_dict(torch.load(adapter_path, map_location=device))
            print("Loaded Adapter weights.")

    @torch.no_grad()
    def generate(self, img, prefix_ids, max_new_tokens=100, temperature=0.7, top_p=0.9, repetition_penalty=1.2):
        """
        Autoregressively generate text given an image and a prefix text.
        """
        # 1. Encode Image
        img_emb, _ = self.qformer.encode_image(img)
        img_emb = self.adapter(img_emb)
        img_emb = img_emb.to(dtype=self.llm.dtype)

        # 2. Encode Prefix Text
        prefix_emb = self.llm.get_input_embeddings()(prefix_ids)

        # 3. Concatenate [Prefix, Image] to form the Prompt
        # Structure matches training: Prefix -> Image -> (Generation starts here)
        inputs_embeds = torch.cat([prefix_emb, img_emb], dim=1)
        
        # 4. Attention Mask
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long)
        
        # 5. Generate
        # We pass inputs_embeds instead of input_ids.
        output_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.llm.config.eos_token_id,
        )
        
        return output_ids

