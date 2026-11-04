"""
Small CNN for single digit classification (designed to stay under 1MB for the web).
"""
import torch.nn as nn


def conv_block(in_channels, out_channels):
    """3x3 합성곱 + 배치 정규화 + ReLU 묶음"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class DigitCNN(nn.Module):
    """
    입력: [N, 1, 28, 28] (검은 배경, 흰 글씨, 0~1)
    출력: [N, 10] 각 숫자(0~9)의 로짓
    """

    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 32),
            nn.MaxPool2d(2),          # 28x28 -> 14x14
            conv_block(32, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2),          # 14x14 -> 7x7
            conv_block(64, 128),
            nn.AdaptiveAvgPool2d(1),  # 7x7 -> 1x1 (전역 평균 풀링으로 파라미터 수 절약)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_parameters(model):
    """학습 가능한 파라미터 개수"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
