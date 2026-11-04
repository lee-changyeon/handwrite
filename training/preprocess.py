"""
Canonical digit normalization shared by training and the web app.

The web app (web/js/preprocess.js) must implement exactly the same steps,
so only simple operations are used here (no library-specific resampling).
"""
from functools import lru_cache

import numpy as np

IMAGE_SIZE = 28      # 모델 입력 크기 (28x28)
DIGIT_BOX = 20       # 숫자의 긴 변을 맞출 크기 (MNIST 방식)
INK_THRESHOLD = 0.1  # 이 값보다 밝은 픽셀을 글씨로 판단


@lru_cache(maxsize=None)
def area_resize_matrix(src, dst):
    """
    길이 src를 길이 dst로 바꾸는 면적 평균(박스 필터) 가중치 행렬 (dst x src).
    출력 픽셀 하나가 덮는 입력 구간의 겹친 비율만큼 평균을 냄.
    """
    matrix = np.zeros((dst, src), dtype=np.float32)
    scale = src / dst
    for i in range(dst):
        start, end = i * scale, (i + 1) * scale
        for j in range(int(np.floor(start)), min(int(np.ceil(end)), src)):
            overlap = min(end, j + 1) - max(start, j)
            if overlap > 0:
                matrix[i, j] = overlap / scale
    return matrix


def round_half_up(value):
    """반올림 (JavaScript의 Math.round와 같은 결과가 나오도록 직접 구현)"""
    return int(np.floor(value + 0.5))


def shift_image(image, dy, dx):
    """이미지를 (dy, dx)만큼 이동. 밖으로 밀려난 부분은 버리고 빈 곳은 0으로 채움"""
    h, w = image.shape
    out = np.zeros_like(image)
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    if src_y0 < src_y1 and src_x0 < src_x1:
        out[src_y0 + dy:src_y1 + dy, src_x0 + dx:src_x1 + dx] = image[src_y0:src_y1, src_x0:src_x1]
    return out


def normalize_digit(image):
    """
    숫자 하나가 담긴 이미지(검은 배경, 흰 글씨, 0~1 실수, 크기 무관)를
    28x28 표준 형식으로 변환:
      1) 글씨 영역만 잘라내기
      2) 비율을 유지하며 긴 변을 20px로 맞추기
      3) 28x28 중앙에 배치
      4) 무게중심을 정중앙으로 이동
    """
    image = np.asarray(image, dtype=np.float32)
    out = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    # 1) 글씨가 있는 행/열의 범위를 찾아 잘라냄
    ink = image > INK_THRESHOLD
    rows = np.flatnonzero(ink.any(axis=1))
    cols = np.flatnonzero(ink.any(axis=0))
    if rows.size == 0:
        return out
    crop = image[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    # 2) 긴 변이 DIGIT_BOX가 되도록 면적 평균으로 크기 변경
    h, w = crop.shape
    scale = DIGIT_BOX / max(h, w)
    new_h = max(1, round_half_up(h * scale))
    new_w = max(1, round_half_up(w * scale))
    resized = area_resize_matrix(h, new_h) @ crop @ area_resize_matrix(w, new_w).T

    # 3) 28x28 캔버스 중앙에 배치
    top, left = (IMAGE_SIZE - new_h) // 2, (IMAGE_SIZE - new_w) // 2
    out[top:top + new_h, left:left + new_w] = resized

    # 4) 무게중심이 이미지 중앙(13.5, 13.5)에 오도록 정수 픽셀만큼 이동
    total = out.sum()
    if total > 0:
        coords = np.arange(IMAGE_SIZE, dtype=np.float32)
        cy = (out.sum(axis=1) * coords).sum() / total
        cx = (out.sum(axis=0) * coords).sum() / total
        center = (IMAGE_SIZE - 1) / 2
        out = shift_image(out, round_half_up(center - cy), round_half_up(center - cx))

    return np.clip(out, 0.0, 1.0)
