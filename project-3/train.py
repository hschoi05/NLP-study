# train.py
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from config import config
from dataset import get_data_loaders
from model import GrokkingTransformer
from analysis import compute_logits_grid, get_key_frequencies, calculate_spectral_losses, get_weight_norm

def main():
    # 1. 설정 및 데이터 로드
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    print(f"Device: {config.device}, Epochs: {config.num_epochs}")
    
    (train_x, train_y), (test_x, test_y), (all_x, all_y), train_indices = get_data_loaders()
    
    # 2. 모델 초기화
    model = GrokkingTransformer(config).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay) # [cite: 157]
    criterion = nn.CrossEntropyLoss()
    
    # 3. 데이터 GPU 이동 (미리 이동해둠)
    train_x_d, train_y_d = train_x.to(config.device), train_y.to(config.device)
    test_x_d, test_y_d = test_x.to(config.device), test_y.to(config.device)
    all_y_grid = all_y.to(config.device)
    
    # 4. 학습 루프
    history = {
        "epochs": [], "train_loss": [], "test_loss": [], 
        "train_acc": [], "test_acc": [], 
        "restricted_loss": [], "excluded_loss": [], "weight_norm": []
    }
    
    print("Starting Training...")
    
    for epoch in range(config.num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        logits = model(train_x_d)
        loss = criterion(logits, train_y_d)
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optimizer.step()
        
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_x_d)
                test_loss = criterion(test_logits, test_y_d).item()
                
                train_acc = (logits.argmax(1) == train_y_d).float().mean().item()
                test_acc = (test_logits.argmax(1) == test_y_d).float().mean().item()
                w_norm = get_weight_norm(model)
                
                history["epochs"].append(epoch)
                history["train_loss"].append(loss.item())
                history["test_loss"].append(test_loss)
                history["train_acc"].append(train_acc)
                history["test_acc"].append(test_acc)
                history["weight_norm"].append(w_norm)
                
                # 분석 (비용 절감을 위해 간헐적 실행)
                if epoch % config.calc_every == 0:
                    keys = get_key_frequencies(model, num_freqs=3)
                    logits_grid = compute_logits_grid(model, all_x, config.device)
                    res_loss, exc_loss = calculate_spectral_losses(
                        logits_grid, train_y_d, all_y_grid, train_indices, keys, config.device
                    )
                    history["restricted_loss"].append(res_loss)
                    history["excluded_loss"].append(exc_loss)
                else:
                    last_res = history["restricted_loss"][-1] if history["restricted_loss"] else 0
                    last_exc = history["excluded_loss"][-1] if history["excluded_loss"] else 0
                    history["restricted_loss"].append(last_res)
                    history["excluded_loss"].append(last_exc)

            if epoch % 5000 == 0:
                print(f"Epoch {epoch:5d} | Test Acc: {test_acc:.4f} | W Norm: {w_norm:.1f}")

    # 5. 결과 저장 및 시각화 (간단 예시)
    plot_results(history)

def plot_results(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history["epochs"], history["train_loss"], label="Train")
    axes[0, 0].plot(history["epochs"], history["test_loss"], label="Test")
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title("Loss (Log Scale)")
    axes[0, 0].legend()
    
    # Accuracy
    axes[0, 1].plot(history["epochs"], history["train_acc"], label="Train")
    axes[0, 1].plot(history["epochs"], history["test_acc"], label="Test")
    axes[0, 1].set_title("Accuracy")
    
    # Spectral Losses
    axes[1, 0].plot(history["epochs"], history["excluded_loss"], label="Excluded", color='green')
    axes[1, 0].plot(history["epochs"], history["restricted_loss"], label="Restricted", color='orange')
    axes[1, 0].set_title("Spectral Losses")
    axes[1, 0].legend()
    
    # Weight Norm
    axes[1, 1].plot(history["epochs"], history["weight_norm"], color='purple')
    axes[1, 1].set_title("Weight Norm")
    
    plt.tight_layout()
    plt.savefig("result_plot.png")
    print("Plot saved as result_plot.png")

if __name__ == "__main__":
    main()