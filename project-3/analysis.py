# analysis.py
import torch
import torch.nn as nn
from config import config

def compute_logits_grid(model, all_x, device):
    """전체 PxP 입력에 대한 Logit 계산"""
    model.eval()
    with torch.no_grad():
        logits = model(all_x.to(device))
    return logits.view(config.p, config.p, config.p)

def get_key_frequencies(model, num_freqs=3):
    """임베딩 행렬에서 상위 주파수 추출"""
    W_E = model.token_embed.weight.detach().cpu()[:config.p]
    fft = torch.fft.fft(W_E, dim=0)
    norms = torch.norm(fft, dim=1)
    
    half_p = config.p // 2
    top_k_indices = torch.topk(norms[1:half_p], k=num_freqs).indices + 1
    return top_k_indices.tolist()

def calculate_spectral_losses(logits_grid, train_labels, all_labels, train_indices, key_freqs, device):
    """Restricted(Full) & Excluded(Train) Loss 계산"""
    p = logits_grid.shape[0]
    fft = torch.fft.fft2(logits_grid, dim=(0, 1))
    
    mask_key = torch.zeros((p, p), dtype=torch.bool, device=device)
    mask_key[0, 0] = True 
    for k in key_freqs:
        mask_key[k, k] = True
        mask_key[p-k, p-k] = True
        
    # Restricted Logits
    fft_res = fft * mask_key.unsqueeze(-1)
    logits_res = torch.fft.ifft2(fft_res, dim=(0, 1)).real
    
    # Excluded Logits
    fft_exc = fft * (~mask_key).unsqueeze(-1)
    logits_exc = torch.fft.ifft2(fft_exc, dim=(0, 1)).real
    
    # Loss Calculation
    loss_res = nn.functional.cross_entropy(
        logits_res.reshape(-1, p), all_labels.to(device)
    ).item()
    
    train_flat_indices = train_indices.to(device)
    logits_exc_flat = logits_exc.reshape(-1, p)[train_flat_indices]
    loss_exc = nn.functional.cross_entropy(
        logits_exc_flat, train_labels.to(device)
    ).item()
    
    return loss_res, loss_exc

def get_weight_norm(model):
    """L2 Norm 합"""
    return sum(p.pow(2).sum().item() for p in model.parameters())