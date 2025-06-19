# 실습8: 신용카드 사기 탐지

<br/>

## 개요 (Overview)
- 불균형 데이터 분류 실습을 위한 합성 신용카드 사기 데이터셋입니다.
- SMOTE, RandomForest, LogisticRegression, 신경망 등 다양한 기법을 사용하여 클래스 불균형을 처리하고 사기 거래를 탐지합니다.

<br/>

## 데이터셋 소개
- **합성 신용카드 사기**: 29개 특성(28개 익명화 + 금액)을 가진 10,000개 거래
- 특성: V1-V28 (익명화된 거래 특성) + Amount (거래 금액)
- 타겟: 이진 분류 (0: 정상, 1: 사기)
- 클래스 불균형: 99.5% 정상 거래, 0.5% 사기 거래

<br/>

## 환경 준비 및 실행법
```bash
# 1. 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 2. 패키지 설치
pip3 install -r requirements.txt

# 3. 실습 코드 실행
python3 credit_card_fraud_detection_kr.py
```

<br/>

## 주요 실습 내용
- 현실적인 사기 패턴을 가진 합성 데이터셋 생성
- 데이터 전처리 및 클래스 불균형 분석
- 불균형 데이터에서의 기준 모델 (RandomForest, LogisticRegression)
- 균형 잡힌 훈련 데이터를 위한 SMOTE 오버샘플링
- 딥러닝: PyTorch 신경망
- 시각화: 클래스 분포, 혼동행렬, ROC 곡선, 정밀도-재현율 곡선
- 비즈니스 통찰 및 비용 분석

<br/>

## 결과 해석
- 기준 모델: 불균형 데이터에서의 성능 비교
- SMOTE 균형 모델: 균형 잡힌 클래스로 개선된 사기 탐지
- 딥러닝: 균형 잡힌 데이터에서의 신경망 성능
- 비즈니스 영향: 거짓 음성 vs 거짓 양성 비용 분석

<br/>

## 주요 학습 포인트
- **불균형 데이터**: 치우친 클래스 분포의 도전 과제 이해
- **SMOTE**: 데이터 균형을 위한 합성 소수 클래스 오버샘플링 기법
- **사기 탐지**: 중요한 정확도 요구사항을 가진 실제 비즈니스 응용
- **비용 분석**: 거짓 양성 vs 거짓 음성의 서로 다른 비용

<br/>

## 비즈니스 맥락
- **정상 (0)**: 합법적인 거래 - 승인 처리
- **사기 (1)**: 사기 거래 - 거부 처리
- **거짓 양성**: 합법적 거래를 사기로 잘못 분류 (고객 불편)
- **거짓 음성**: 사기 거래를 놓침 (금융 손실)

<br/>

## 기술적 특징
- **SMOTE**: 소수 클래스를 위한 합성 샘플 생성
- **ROC 곡선**: 모델의 판별 능력 평가
- **정밀도-재현율 곡선**: 불균형 데이터셋에 중요한 지표
- **특성 중요도**: 주요 사기 지표 식별

<br/>

## 마무리 (Conclusion)
- 사기 탐지를 위한 현실적인 불균형 데이터셋으로 실습
- 기준 vs 균형 모델 성능 비교
- 불균형 데이터에 대한 적절한 평가 지표의 중요성 이해
- 사기 탐지 시스템의 비즈니스 영향 학습

<br/>

## 참고 (Reference)
- [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)
- [Credit Card Fraud Detection Best Practices](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Handling Imbalanced Data](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)
- [ROC Curve Analysis](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) 