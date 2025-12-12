from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

image = Image.open("dog.jpg")
model = "google/vit-base-patch16-224"
feature_extractor = ViTImageProcessor.from_pretrained(model)
model = ViTModel.from_pretrained(model)
inputs = feature_extractor(images=image, return_tensors="pt")
print(inputs.pixel_values.shape)
with torch.no_grad():
    outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
