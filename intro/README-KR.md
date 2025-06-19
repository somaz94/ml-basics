# intro: 머신러닝 기초 예제

이 디렉토리는 머신러닝의 대표적인 기초 개념(지도학습, 비지도학습, 유사도 비교)을 한 파일에 통합한 실습 예제를 제공합니다.

<br/>

## 예제 파일
- `fruit_ml_examples_kr.py`: DecisionTree, KMeans, Cosine Similarity를 활용한 과일 데이터 분류/군집/유사도 실습

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
python3 fruit_ml_examples_kr.py
```

<br/>

## 주요 내용
- **지도학습**: DecisionTreeClassifier로 과일 분류
- **비지도학습**: KMeans로 과일 군집화 및 시각화
- **유사도 비교**: Cosine Similarity로 과일 간 유사도 계산

<br/>

## 참고
- 각 블록별로 코드를 분리하거나, 다양한 데이터/모델로 확장해볼 수 있습니다. 