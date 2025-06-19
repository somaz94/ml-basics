# 실습6: MNIST 손글씨 숫자 분류

<br/>

## 개요 (Overview)
- MNIST 손글씨 숫자 데이터셋을 활용한 지도학습(분류), 딥러닝(CNN), 비지도학습(군집화), 시각화 실습입니다.
- scikit-learn의 fetch_openml로 데이터셋을 불러오며, RandomForest, SVM, CNN, KMeans 등 다양한 알고리즘을 적용합니다.

<br/>

## 데이터셋 소개
- **MNIST**: 28x28 픽셀 해상도의 70,000개 손글씨 숫자 이미지 (0-9)
- 출처: [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- 특성: 784개 픽셀 값 (28×28 평면화)
- 타겟: 숫자 라벨 (0-9)

<br/>

## 환경 준비 및 실행법
```bash
# 1. 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 실습 코드 실행
python mnist_digit_classification_kr.py
```

<br/>

## 주요 실습 내용
- 데이터 로드 및 전처리 (fetch_openml, 데이터 재구성)
- 지도학습: RandomForest, SVM으로 숫자 분류, 정확도 비교
- 딥러닝: CNN 아키텍처, 학습, 평가
- 비지도학습: 10개 군집으로 KMeans 군집화
- 시각화: 샘플 숫자 이미지, 혼동행렬, 군집 결과
- 모델 비교 및 성능 분석

<br/>

## 결과 해석
- 지도학습: RandomForest, SVM, CNN 간 정확도 비교
- 딥러닝: 공간적 특성 학습으로 CNN이 일반적으로 더 높은 정확도 달성
- 비지도학습: 라벨 없이 KMeans가 유사한 숫자 패턴을 그룹화
- 시각화: 혼동행렬로 자주 잘못 분류되는 숫자 확인

<br/>

## 주요 학습 포인트
- **이미지 데이터 처리**: CNN을 위한 1D 배열을 2D 이미지로 재구성
- **딥러닝 기초**: 합성곱층, 풀링, 드롭아웃, 밀집층
- **모델 비교**: 전통적 ML vs 딥러닝 성능
- **컴퓨터 비전**: 이미지 데이터의 공간적 특성 이해

<br/>

## 마무리 (Conclusion)
- 컴퓨터 비전의 가장 유명한 데이터셋으로 실습
- 전통적 머신러닝과 딥러닝 접근법 비교
- 이미지 분류 작업에서 CNN의 강력함 이해
- 이미지 데이터 전처리 기법 학습

<br/>

## 참고 (Reference)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow MNIST Tutorial](https://www.tensorflow.org/tutorials/quickstart/beginner)
- [Scikit-learn MNIST](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html)
- [Keras CNN Guide](https://keras.io/examples/vision/mnist_convnet/) 