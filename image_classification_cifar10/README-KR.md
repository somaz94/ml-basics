# 이미지 분류 (CIFAR-10) (실습12)

- **주제:** 컬러 이미지 분류 실습
- **예제:** CNN(Convolutional Neural Network), 데이터 증강
- **특징:** 딥러닝 컴퓨터 비전 기초

## 실행법

```bash
cd image_classification_cifar10
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 image_classification_cifar10_kr.py
```

## 주요 내용
- torchvision에서 CIFAR-10 데이터셋 자동 다운로드
- 데이터 증강(Data Augmentation)을 통한 모델 성능 향상
- PyTorch를 사용한 CNN 모델 설계 및 학습
- 학습 과정 시각화 및 모델 평가
- 예측 결과 이미지와 함께 확인 