import os
import torch

# 1. Project Base Directory
# src/config.py의 상위(src)의 상위(project-4)를 BASE_DIR로 잡음
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Data Paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')      # 검색할 이미지가 있는 폴더
INDEX_PATH = os.path.join(DATA_DIR, 'embeddings.pt') # 임베딩 저장 파일 경로

# 3. Model
MODEL_NAME = "openai/clip-vit-base-patch32" # 사용할 CLIP 모델

# 4. Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure data directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)