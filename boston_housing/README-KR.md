# 실습5: 보스턴 집값 예측 (Boston Housing)

<br/>

## 개요 (Overview)
- 보스턴 주택 가격 데이터셋을 활용한 지도학습(회귀), 비지도학습(군집화), 시각화 실습입니다.
- scikit-learn의 fetch_openml로 데이터셋을 불러오며, LinearRegression, RandomForest, KMeans 등 다양한 알고리즘을 적용합니다.

<br/>

## 데이터셋 소개
- **Boston Housing**: 1970년대 미국 보스턴 지역의 주택 가격과 13개 특성(방 개수, 범죄율 등) 데이터
- 출처: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Housing)

<br/>

## 환경 준비 및 실행법
```bash
# 1. 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 2. 패키지 설치
pip3 install -r requirements.txt

# 3. 실습 코드 실행
python3 boston_housing_kr.py
```

<br/>

## 주요 실습 내용
- 데이터 로드 및 전처리 (fetch_openml, pandas)
- 지도학습: LinearRegression, RandomForestRegressor로 집값 예측, MSE 출력
- 비지도학습: KMeans로 지역 군집화, 군집별 샘플 수 출력
- 시각화: 방 개수(RM)-집값(MEDV) 산점도, 군집 결과 시각화
- 한글 폰트 설정(AppleGothic)

<br/>

## 결과 해석
- 지도학습: MSE(평균제곱오차)로 회귀 성능 비교
- 비지도학습: KMeans로 3개 군집 분류, 각 군집별 특성 확인 가능
- 시각화: 방 개수 증가에 따라 집값이 증가하는 경향 확인

<br/>

## 마무리 (Conclusion)
- 다양한 회귀/군집화 알고리즘을 실습하며, 특성-타겟 관계와 군집별 특성 차이를 시각적으로 이해할 수 있습니다.

<br/>

## 참고 (Reference)
- [scikit-learn Boston Housing](https://scikit-learn.org/stable/datasets/toy_dataset.html#boston-house-prices-dataset)
- [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Housing)
- [Kaggle Boston Housing](https://www.kaggle.com/datasets/altavish/boston-housing-dataset) 