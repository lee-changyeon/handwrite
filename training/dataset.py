"""
Load MNIST + EMNIST(digits), normalize them to the canonical format and cache the result.
Run directly to download and prepare the data: python dataset.py
"""
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

from preprocess import normalize_digit

DATA_DIR = Path(__file__).parent / "data"
VAL_SIZE = 10_000  # 학습 데이터 중 검증용으로 떼어 둘 개수


def _load_raw(name, train):
    """torchvision으로 원본 데이터를 받아 (이미지 uint8 [N,28,28], 라벨 int64) 반환"""
    if name == "mnist":
        ds = datasets.MNIST(DATA_DIR, train=train, download=True)
        images = ds.data
    else:
        ds = datasets.EMNIST(DATA_DIR, split="digits", train=train, download=True)
        # EMNIST 이미지는 가로세로가 뒤바뀌어(전치) 저장되어 있어서 바로잡음
        images = ds.data.transpose(1, 2)
    return images.numpy(), ds.targets.numpy().astype(np.int64)


def _normalize_all(images):
    """모든 이미지를 표준 형식으로 변환하여 uint8로 반환 (저장 용량 절약)"""
    out = np.empty_like(images)
    for i, img in enumerate(images):
        out[i] = np.round(normalize_digit(img / 255.0) * 255).astype(np.uint8)
    return out


def load_split(name, train):
    """정규화된 데이터를 캐시에서 읽고, 없으면 새로 만들어 저장"""
    tag = "train" if train else "test"
    cache = DATA_DIR / f"canonical_{name}_{tag}.npz"
    if cache.exists():
        data = np.load(cache)
        return data["images"], data["labels"]

    images, labels = _load_raw(name, train)
    start = time.time()
    print(f"Normalizing {name} {tag} ({len(images):,} images) ...")
    images = _normalize_all(images)
    print(f"  done in {time.time() - start:.0f}s")
    DATA_DIR.mkdir(exist_ok=True)
    np.savez_compressed(cache, images=images, labels=labels)
    return images, labels


def load_all(seed=0):
    """학습/검증/테스트 데이터를 모두 불러와 텐서(uint8 이미지, int64 라벨)로 반환"""
    mnist_x, mnist_y = load_split("mnist", train=True)
    emnist_x, emnist_y = load_split("emnist", train=True)

    # MNIST와 EMNIST 학습 데이터를 합친 뒤 섞어서 일부를 검증용으로 분리
    x = np.concatenate([mnist_x, emnist_x])
    y = np.concatenate([mnist_y, emnist_y])
    order = np.random.default_rng(seed).permutation(len(x))
    x, y = x[order], y[order]

    splits = {
        "train": (x[VAL_SIZE:], y[VAL_SIZE:]),
        "val": (x[:VAL_SIZE], y[:VAL_SIZE]),
        "test_mnist": load_split("mnist", train=False),
        "test_emnist": load_split("emnist", train=False),
    }
    return {k: (torch.from_numpy(a), torch.from_numpy(b)) for k, (a, b) in splits.items()}


def to_float(images_uint8):
    """uint8 [N,28,28] 이미지를 모델 입력 형식 float [N,1,28,28] (0~1)로 변환"""
    return images_uint8.unsqueeze(1).float().div_(255.0)


def augment(x, generator=None):
    """
    데이터 증강: 이미지마다 서로 다른 무작위 변형을 적용하여 다양한 필체에 대응.
    x: float [N,1,28,28]
    """
    n = x.shape[0]

    def uniform(low, high):
        return torch.empty(n).uniform_(low, high, generator=generator)

    # 회전, 크기, 기울이기(shear), 이동을 합친 아핀 변환 행렬을 이미지마다 생성
    angle = uniform(-12, 12) * math.pi / 180
    scale = uniform(0.85, 1.15)
    shear = uniform(-0.2, 0.2)
    tx, ty = uniform(-0.1, 0.1), uniform(-0.1, 0.1)  # 정규화 좌표 기준 (약 ±1.4px)

    cos, sin = torch.cos(angle), torch.sin(angle)
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = (cos + shear * sin) / scale
    theta[:, 0, 1] = (-sin + shear * cos) / scale
    theta[:, 1, 0] = sin / scale
    theta[:, 1, 1] = cos / scale
    theta[:, 0, 2], theta[:, 1, 2] = tx, ty

    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # 획 두께 변화: 일부는 굵게(팽창), 일부는 가늘게(가장자리 약화)
    r = torch.rand(n, generator=generator).view(n, 1, 1, 1)
    thick = torch.maximum(x, 0.7 * F.max_pool2d(x, 3, stride=1, padding=1))
    thin = x * x
    x = torch.where(r < 0.15, thick, torch.where(r < 0.30, thin, x))
    return x.clamp_(0.0, 1.0)


if __name__ == "__main__":
    # 데이터 다운로드와 정규화만 미리 실행
    data = load_all()
    for split, (images, labels) in data.items():
        print(f"{split:12s} {tuple(images.shape)}  labels {torch.bincount(labels).tolist()}")
