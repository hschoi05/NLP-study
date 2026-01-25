import os
import torch
import requests
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from src.config import IMAGE_DIR, INDEX_PATH
from src.model import CLIPHandler
from datasets import load_dataset

class SearchEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip = CLIPHandler(self.device)
        self.image_paths = []
        self.embeddings = None

        if not os.path.exists(IMAGE_DIR):
            os.makedirs(IMAGE_DIR)
        
        if not os.listdir(IMAGE_DIR):
            print("Local images not found. Downloading MS COCO subset...")
            self.download_from_urls(num_images=200)
        
        # Load or Build Index
        if os.path.exists(INDEX_PATH):
            self.load_index()
        else:
            print("Index not found. Building index...")
            self.build_index()

    def download_from_urls(self, num_images=200):
        """
        Download images from a predefined list of URLs
        save in data/images folder
        """
        num_images = 200

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        print(f"Downloading {num_images} images from predefined URLs...")
        
        success_count = 0
        for i in tqdm(range(num_images)):
            try:
                url = f"https://picsum.photos/600/400?random={i}"
                
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                img = Image.open(BytesIO(response.content)).convert("RGB")

                save_path = os.path.join(IMAGE_DIR, f"random_{i}.jpg")
                img.save(save_path)
                success_count += 1
           
            except Exception as e:
                print(f"Failed to download image from {url}: {e}")
        
        print(f"Downloaded {success_count} images to {IMAGE_DIR}")


    def download_coco_subset(self, num_images=200):
        """
        Download COCO 2017 VAl dataset from Huggingface
        save in data/images folder
        """
        try:
            dataset = load_dataset("merve/coco2017", split="validation", streaming=True)
            print(f"Downloading {num_images} images from MS COCO (via Hugging Face)...")
            count = 0
            for item in tqdm(dataset, total=num_images):
                if count >= num_images:
                    break

                # Debugging
                if count == 0:
                    print(f"Data column list: {list(item.keys())}")
                
                image = None
                if 'image' in item:
                    image = item['image']
                elif 'img' in item:
                    image = item['img']
                elif 'pixel_values'in item:
                    image = item['pixel_values']

                if image is None:
                    print("No image found in the dataset item!")
                    continue

                save_path = os.path.join(IMAGE_DIR, f"coco_{count}.jpg")
                image.convert("RGB").save(save_path)
                count += 1
            print(f"Downloaded {count} images to {IMAGE_DIR}")
        
        except Exception as e:
            print(f"Error downloading COCO dataset: {e}")

    def build_index(self):
        """data/images 폴더의 모든 이미지를 인코딩하여 저장"""
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        self.image_paths = [
            os.path.join(IMAGE_DIR, f) 
            for f in os.listdir(IMAGE_DIR) 
            if f.lower().endswith(valid_extensions)
        ]

        if not self.image_paths:
            print("No images found in data/images!")
            return

        all_embeddings = []
        batch_size = 32

        print(f"Indexing {len(self.image_paths)} images...")
        
        for i in tqdm(range(0, len(self.image_paths), batch_size)):
            batch_paths = self.image_paths[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            
            # (B, 512)
            emb = self.clip.get_image_embedding(images)
            all_embeddings.append(emb.cpu())

        # Concatenate all batches
        self.embeddings = torch.cat(all_embeddings, dim=0)
        # self.embeddings: (N_images, 512)

        # Save to disk
        data = {'paths': self.image_paths, 'embeddings': self.embeddings}
        torch.save(data, INDEX_PATH)
        print(f"Saved index to {INDEX_PATH}")

    def load_index(self):
        print(f"Loading index from {INDEX_PATH}...")
        data = torch.load(INDEX_PATH)
        self.image_paths = data['paths']
        self.embeddings = data['embeddings'].to(self.device)
        print(f"Loaded {len(self.image_paths)} images.")

    def search(self, query, top_k=5, mode='text'):
        """
        query: str (text) or PIL.Image (image)
        mode: 'text' or 'image'
        """
        # 1. Encode Query
        if mode == 'text':
            query_emb = self.clip.get_text_embedding([query]) 
            # query_emb: (1, 512)
        else:
            query_emb = self.clip.get_image_embedding([query])
            # query_emb: (1, 512)

        # 2. Compute Cosine Similarity (Dot product since normalized)
        # (1, 512) @ (512, N) -> (1, N) -> (N,)
        scores = torch.matmul(query_emb, self.embeddings.T).squeeze(0)

        # 3. Retrieve Top-K
        top_scores, top_indices = torch.topk(scores, k=min(top_k, len(self.embeddings)))
        
        results = []
        for score, idx in zip(top_scores, top_indices):
            results.append({
                "path": self.image_paths[idx.item()],
                "score": score.item()
            })
        
        return results