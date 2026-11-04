"""
Train the digit CNN on MNIST + EMNIST(digits).
Usage: python train.py [--epochs 8] [--batch-size 256] [--lr 3e-3]
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from dataset import augment, load_all, to_float
from model import DigitCNN, count_parameters

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "digit_cnn.pt"
TARGET_MNIST_ACC = 99.3  # M1 완료 기준 (MNIST 테스트 정확도, %)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the digit CNN")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, images, labels, batch_size=2000):
    """정확도(%) 계산 (평가 모드: 드롭아웃 끄고 배치 정규화는 저장된 통계 사용)"""
    model.eval()
    correct = 0
    for i in range(0, len(images), batch_size):
        logits = model(to_float(images[i:i + batch_size]))
        correct += (logits.argmax(dim=1) == labels[i:i + batch_size]).sum().item()
    return 100.0 * correct / len(images)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    # 1) 데이터 불러오기
    data = load_all()
    train_x, train_y = data["train"]
    print(f"Train {len(train_x):,} | Val {len(data['val'][0]):,} | "
          f"MNIST test {len(data['test_mnist'][0]):,} | EMNIST test {len(data['test_emnist'][0]):,}")

    # 2) 모델, 손실 함수, 옵티마이저, 학습률 스케줄러 준비
    model = DigitCNN()
    print(f"Model parameters: {count_parameters(model):,} "
          f"(~{count_parameters(model) * 4 / 1024:.0f} KB as float32)")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # 과신 방지를 위한 라벨 스무딩
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = (len(train_x) + args.batch_size - 1) // args.batch_size
    # OneCycle: 학습률을 처음엔 올렸다가 점점 낮춰 빠르고 안정적으로 수렴
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_val, history = 0.0, []
    train_start = time.time()

    # 3) 학습 반복
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        order = torch.randperm(len(train_x), generator=generator)
        total_loss = 0.0

        for step in range(steps_per_epoch):
            idx = order[step * args.batch_size:(step + 1) * args.batch_size]
            x = augment(to_float(train_x[idx]), generator)
            loss = criterion(model(x), train_y[idx])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            # 진행 상황과 남은 시간을 주기적으로 출력
            if (step + 1) % 200 == 0:
                elapsed = time.time() - epoch_start
                eta = elapsed / (step + 1) * (steps_per_epoch - step - 1)
                print(f"  epoch {epoch} step {step + 1}/{steps_per_epoch} "
                      f"loss {total_loss / (step + 1):.4f} | epoch ETA {eta / 60:.1f} min", flush=True)

        # 4) 에폭마다 평가하고, 검증 정확도가 가장 좋은 모델을 저장
        metrics = {
            "epoch": epoch,
            "loss": round(total_loss / steps_per_epoch, 4),
            "val": round(evaluate(model, *data["val"]), 2),
            "mnist_test": round(evaluate(model, *data["test_mnist"]), 2),
            "emnist_test": round(evaluate(model, *data["test_emnist"]), 2),
            "minutes": round((time.time() - epoch_start) / 60, 1),
        }
        history.append(metrics)
        saved = ""
        if metrics["val"] > best_val:
            best_val = metrics["val"]
            torch.save({"model": model.state_dict(), "metrics": metrics}, CHECKPOINT_PATH)
            saved = " (saved)"
        print(f"Epoch {epoch}/{args.epochs} | loss {metrics['loss']:.4f} | val {metrics['val']:.2f}% | "
              f"MNIST {metrics['mnist_test']:.2f}% | EMNIST {metrics['emnist_test']:.2f}% | "
              f"{metrics['minutes']} min{saved}", flush=True)

    # 5) 최종 결과 정리
    (CHECKPOINT_DIR / "history.json").write_text(json.dumps(history, indent=2))
    best = torch.load(CHECKPOINT_PATH)["metrics"]
    passed = best["mnist_test"] >= TARGET_MNIST_ACC
    print(f"\nBest epoch {best['epoch']}: MNIST {best['mnist_test']:.2f}% | EMNIST {best['emnist_test']:.2f}% "
          f"| total {(time.time() - train_start) / 60:.0f} min")
    print(f"M1 target (MNIST >= {TARGET_MNIST_ACC}%): {'PASSED' if passed else 'NOT MET'}")


if __name__ == "__main__":
    main()
