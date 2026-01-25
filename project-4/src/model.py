import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from src.config import MODEL_NAME

class CLIPHandler:
    def __init__(self, device='cpu'):
        self.device = device
        # print(f"Loading CLIP model: {MODEL_NAME} on {self.device}...")
        self.model = CLIPModel.from_pretrained(MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        self.model.eval()

    def get_image_embedding(self, images):
        """
        Convert image list into normalized embeddings
        images: List[PIL.Image] or PIL.Image
        """
        # Preprocess
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        # inputs['pixel_values']: (B, 3, 224, 224)
        # pt: pytorch tensor


        with torch.no_grad():
            # (B, 3, 224, 224) -> (B, 512) for ViT-B/32
            # ViT, Projection
            outputs = self.model.get_image_features(**inputs)

        # L2 Normalize: Make length 1 (p=2, dim=-1)
        # (B, 512) / (B, 1) -> (B, 512)
        embeddings = outputs / outputs.norm(p=2, dim=-1, keepdim=True)

        return embeddings

    def get_text_embedding(self, text):
        """
        Convert text(str or List[str]) into normalized embeddings
        """
        # Preprocess
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        # inputs['input_ids']: (B, 77)
        # inputs['attention_mask']: (B, 77)
        # truncation: text 길이 77 초과 시 자름

        with torch.no_grad():
            # (B, 77) -> (B, 512) for ViT-B/32
            outputs = self.model.get_text_features(**inputs)

        # L2 Normalize
        embeddings = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        # embeddings: (B, 512)

        return embeddings