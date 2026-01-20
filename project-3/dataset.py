# dataset.py
import torch
from config import config

def create_dataset_strict(p):
    """
    [a, b, =] 형태의 입력과 (a+b)%p 라벨을 생성합니다.
    [cite_start]마지막 토큰 '='의 인덱스는 p입니다. [cite: 16]
    """
    a_vals = torch.arange(p)
    b_vals = torch.arange(p)
    grid_a, grid_b = torch.meshgrid(a_vals, b_vals, indexing='ij')
    
    # (p*p, 2)
    pairs = torch.stack([grid_a.flatten(), grid_b.flatten()], dim=1)
    # (p*p, 1) -> 구분자 토큰
    eq_token = torch.full((p*p, 1), p, dtype=torch.long)
    
    # 입력: [a, b, =]
    inputs = torch.cat([pairs, eq_token], dim=1)
    # 라벨: (a + b) % p
    labels = (pairs[:, 0] + pairs[:, 1]) % p
    
    return inputs, labels

def get_data_loaders():
    torch.manual_seed(config.seed)
    
    all_x, all_y = create_dataset_strict(config.p)
    
    # 데이터 분할
    perm = torch.randperm(len(all_x))
    split_idx = int(len(all_x) * config.train_fraction)
    
    train_indices = perm[:split_idx]
    test_indices = perm[split_idx:]
    
    train_x, train_y = all_x[train_indices], all_y[train_indices]
    test_x, test_y = all_x[test_indices], all_y[test_indices]
    
    return (train_x, train_y), (test_x, test_y), (all_x, all_y), train_indices