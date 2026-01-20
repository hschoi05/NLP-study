# config.py
import torch

class Config:
    def __init__(self):
        self.p = 113              # 소수 모듈러스
        self.d_model = 128        # 임베딩 차원
        self.num_heads = 4        # Attention Head 수
        self.d_mlp = 512          # MLP 히든 차원
        self.lr = 1e-3            # 학습률
        [cite_start]self.weight_decay = 1.0   # Grokking 핵심: 높은 가중치 감쇠 [cite: 149, 157, 358]
        [cite_start]self.num_epochs = 40000   # 충분한 학습 시간 (Cleanup 단계 확인용) [cite: 27, 158, 359]
        [cite_start]self.train_fraction = 0.3 # 훈련 데이터 비율 [cite: 156, 357]
        self.seed = 42
        self.seq_len = 3          # [a, b, =] 구조
        [cite_start]self.clip_grad = 1.0      # 학습 불안정 방지 [cite: 358]
        self.calc_every = 500     # 분석 메트릭 계산 주기
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()