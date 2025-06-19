# 실습4: 타이타닉 생존자 예측/군집화 실습

이 디렉토리는 타이타닉 데이터셋을 활용한 지도학습(생존 예측), 비지도학습(군집화), 시각화 실습 예제를 제공합니다.

<br/>

## 예제 파일
- `titanic_survival_kr.py`: DecisionTree, RandomForest, LogisticRegression으로 생존 예측, KMeans로 승객 군집화, 산점도 시각화

<br/>

## 실행 방법

1. 가상환경 생성 및 활성화 (권장)
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. 의존성 설치
```bash
pip3 install -r requirements.txt
```

3. 예제 실행
```bash
python3 titanic_survival_kr.py
```

<br/>


## 주요 내용
- **지도학습**: DecisionTree, RandomForest, LogisticRegression으로 생존 예측 및 정확도 비교
- **비지도학습**: KMeans로 승객 데이터 군집화, 실제 생존자와 비교
- **시각화**: 나이-요금-생존 산점도, 군집 결과 시각화

<br/>

## 참고
- 다양한 특성 엔지니어링, 결측치 처리, 시각화 방법으로 확장해볼 수 있습니다. 