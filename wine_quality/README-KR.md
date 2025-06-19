# 실습3: 와인 품질 데이터 분류/회귀/군집화 실습

이 디렉토리는 와인 품질 데이터셋을 활용한 지도학습(분류/회귀), 비지도학습(군집화), 시각화 실습 예제를 제공합니다.

## 예제 파일
- `wine_quality_kr.py`: RandomForest로 품질 등급 분류, LinearRegression으로 품질 점수 예측, KMeans로 군집화, 산점도 시각화

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
python3 wine_quality_kr.py
```

<br/>

## 주요 내용
- **지도학습**: RandomForest로 품질 등급 분류, LinearRegression으로 품질 점수 예측
- **비지도학습**: KMeans로 와인 데이터 군집화
- **시각화**: 알코올-산도-품질 산점도, 군집 결과 시각화

<br/>

## 참고
- 다양한 분류/회귀/군집 알고리즘, 시각화 방법으로 확장해볼 수 있습니다. 