# model.py
import torch
import torch.nn as nn

class GrokkingTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.p = cfg.p
        [cite_start]self.vocab_size = cfg.p + 1 # 0~112 + '='(113) [cite: 156]
        self.seq_len = cfg.seq_len
        
        # 임베딩 & 위치 인코딩
        self.token_embed = nn.Embedding(self.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(self.seq_len, cfg.d_model)
        
        # [cite_start]역공학을 위해 bias=False 설정 [cite: 355]
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.num_heads, batch_first=True, bias=False
        )
        self.mlp_in = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.mlp_out = nn.Linear(cfg.d_mlp, cfg.d_model, bias=False)
        self.relu = nn.ReLU()
        self.unembed = nn.Linear(cfg.d_model, cfg.p, bias=False)

    def forward(self, x):
        positions = torch.arange(self.seq_len, device=x.device)
        h = self.token_embed(x) + self.pos_embed(positions)
        
        # Attention
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = h + attn_out
        
        # MLP
        mlp_pre = self.mlp_in(h)
        mlp_act = self.relu(mlp_pre)
        mlp_out = self.mlp_out(mlp_act)
        h = h + mlp_out
        
        # [cite_start]마지막 토큰(=) 위치의 벡터로 예측 [cite: 156]
        logits = self.unembed(h[:, -1, :])
        return logits