# 실습7: 유방암 진단

<br/>

## 개요 (Overview)
- 유방암 위스콘신 데이터셋을 활용한 의료 AI 분류 및 군집화 실습입니다.
- scikit-learn의 load_breast_cancer로 데이터셋을 불러오며, LogisticRegression, RandomForest, 신경망, KMeans 등 다양한 알고리즘을 적용합니다.

<br/>

## 데이터셋 소개
- **유방암 위스콘신**: 유방암 진단을 위한 569개 샘플과 30개 특성
- 출처: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- 특성: 세포 핵 특성에서 계산된 30개 수치형 특성
- 타겟: 이진 분류 (0: 양성, 1: 악성)

<br/>

## 환경 준비 및 실행법
```bash
# 1. 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 실습 코드 실행
python breast_cancer_diagnosis_kr.py
```

<br/>

## 주요 실습 내용
- 데이터 로드 및 전처리 (StandardScaler, train_test_split)
- 지도학습: LogisticRegression, RandomForest로 암 분류
- 딥러닝: PyTorch 신경망으로 의료 진단
- 비지도학습: 2개 군집으로 KMeans 군집화
- 시각화: 특성 중요도, 혼동행렬, ROC 곡선, PCA 군집화
- 의료적 해석 및 임상적 권장사항

<br/>

## 결과 해석
- 지도학습: LogisticRegression, RandomForest, 신경망 간 정확도 및 AUC 비교
- 딥러닝: 의료 진단에서 신경망이 일반적으로 높은 정확도 달성
- 비지도학습: 라벨 없이 KMeans가 유사한 암 패턴을 그룹화
- 의료적 통찰: 특성 중요도로 주요 진단 특성 확인

<br/>

## 주요 학습 포인트
- **의료 AI**: 의료 응용에서 정확도의 중요성 이해
- **특성 공학**: 중요한 진단 특성 식별
- **모델 평가**: 여러 지표(정확도, AUC, 혼동행렬) 사용
- **임상 응용**: 실제 의료 진단 시나리오

<br/>

## 의료적 맥락
- **양성 (0)**: 비암성 종양, 일반적으로 생명에 위험하지 않음
- **악성 (1)**: 암성 종양, 즉시 치료가 필요
- **거짓 음성**: 의료 진단에서 거짓 양성보다 더 위험
- **특성 중요도**: 어떤 세포 특성이 가장 진단에 중요한지 이해

<br/>

## 마무리 (Conclusion)
- 암 진단을 위한 실제 의료 데이터로 실습
- 전통적 머신러닝과 딥러닝 접근법 비교
- 의료 AI에서 정확도의 중요성 이해
- 임상적 의사결정을 위한 특성 중요도 분석 학습

<br/>

## 참고 (Reference)
- [UCI Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [Scikit-learn Breast Cancer](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)
- [Medical AI Best Practices](https://www.nature.com/articles/s41591-019-0648-5)
- [ROC Curve Analysis](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) 