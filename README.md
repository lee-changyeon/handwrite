# ✍️ Handwritten Number Recognizer (손글씨 숫자 인식기)

웹 브라우저에서 여러 자리 숫자를 손으로 쓰면 인식해 주는 프로그램입니다.
설치 없이 링크만 열면 PC와 휴대폰에서 바로 쓸 수 있습니다.

> 🚧 **개발 중입니다** (M0 저장소 준비 완료). 개발 계획과 진행 상황은 [PLAN.md](PLAN.md)를 참고하세요.

**🔗 데모:** https://lee-changyeon.github.io/handwrite/ (M6 배포 후 동작 예정)

## 주요 기능
- 마우스, 터치, 펜으로 여러 자리 숫자 입력 (예: `2026`)
- 숫자별 인식 결과와 확신도 표시, 자신 없는 숫자는 강조
- 되돌리기, 지우기, 결과 복사
- **모든 인식이 브라우저 안에서 실행**되어 그린 그림이 서버로 전송되지 않음

## 동작 원리
1. 캔버스에 쓴 획을 가로 위치 기준으로 묶어 숫자 하나씩 분리합니다.
2. 각 숫자를 학습 데이터와 같은 28x28 형식으로 변환합니다.
3. CNN 모델(MNIST + EMNIST로 학습, ONNX 형식)이 브라우저에서 각 숫자를 분류합니다.

## 사용 팁
- 숫자 사이를 **조금씩 띄워서** 써 주세요. 붙여 쓴 숫자는 아직 지원하지 않습니다.
- 현재는 0~9 숫자만 인식합니다.

## 로컬에서 실행하기

### 저장소 받기
```bash
git clone https://github.com/lee-changyeon/handwrite.git
cd handwrite
```

### 웹 앱 실행
브라우저 보안 정책 때문에 `index.html`을 더블클릭해서 열면 모델이 로드되지 않습니다. 로컬 서버로 열어 주세요.
```bash
cd web
python -m http.server 8000
# 브라우저에서 http://localhost:8000 접속
```

### 모델 다시 학습하기 (선택)
```bash
cd training
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python train.py                # MNIST + EMNIST 다운로드 후 학습
python export_onnx.py          # web/model/digit_cnn.onnx 생성
```

## 폴더 구조
```
training/   모델 학습과 ONNX 변환 (Python, PyTorch)
web/        GitHub Pages로 배포되는 웹 앱 (HTML, CSS, JavaScript)
tests/      숫자 분리와 전처리 테스트
```

## 기술 스택
- **학습:** Python, PyTorch, torchvision
- **추론:** ONNX, onnxruntime-web
- **화면:** HTML Canvas, 순수 JavaScript
- **배포:** GitHub Pages, GitHub Actions

## 라이선스
[MIT](LICENSE)
