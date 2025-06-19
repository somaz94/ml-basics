# 실습2: 붓꽃(Iris) 데이터 분류/군집화 실습

이 디렉토리는 Iris(붓꽃) 데이터셋을 활용한 지도학습(분류), 비지도학습(군집화), 2D/3D 시각화 실습 예제를 제공합니다.

<br/>

## 예제 파일
- `iris_classification.py`: SVM, KNN, LogisticRegression으로 품종 분류, KMeans로 군집화 및 실제 품종과 비교, 2D/3D 시각화

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
python3 iris_classification.py
```

<br/>

## 주요 내용
- **지도학습**: SVM, KNN, LogisticRegression으로 붓꽃 품종 분류 및 정확도 비교
- **비지도학습**: KMeans로 군집화, 실제 품종과 군집 결과 비교(혼동 행렬)
- **시각화**: 2D/3D scatter plot으로 실제 품종/군집 결과 시각화

<br/>

## 참고
- 다양한 분류/군집 알고리즘, 시각화 방법으로 확장해볼 수 있습니다. 