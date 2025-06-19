# 주식 가격 예측 (실습10)

- **주제:** 시계열 데이터 회귀 실습
- **예제:** LinearRegression, LSTM, 시계열 시각화
- **특징:** 시계열 분석 기초, 실제 주가 데이터 활용

---

## 실습 구성

1. **미국 주식(Yahoo Finance) 기반 실습**
   - 글로벌 대표 주식(예: 애플) 데이터로 시계열 회귀 실습
   - 데이터 소스: Yahoo Finance
   - 파일: `stock_price_prediction_kr.py`

2. **국내 코스피 지수(네이버 금융) 기반 실습**
   - 네이버 금융에서 코스피(KOSPI) 지수 일별 시세 크롤링
   - 국내 시계열 데이터 분석 및 예측 실습
   - 파일: `stock_price_prediction_kr_naver.py`

---

## 실행법

### 1. 미국 주식(Yahoo Finance) 실습
```bash
cd stock_price_prediction
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 stock_price_prediction_kr.py
```

### 2. 코스피(네이버 금융) 실습
```bash
cd stock_price_prediction
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 stock_price_prediction_kr_naver.py
```

---

## 주요 내용 및 실무 팁
- **실제 데이터 활용:** 글로벌(미국)과 국내(코스피) 시계열 데이터 모두 실습 가능
- **모델 비교:** 선형회귀(LinearRegression)와 LSTM 딥러닝 모델의 예측력 비교
- **시계열 분석:** 데이터 윈도우, 시계열 특성 생성, 시각화, 과적합 주의 등 실무 적용 팁 제공
- **크롤링 실습:** 네이버 금융 데이터는 웹 크롤링으로 수집하며, 결측치/이상치 처리 필요
- **확장성:** 다양한 종목, 지수, 윈도우 크기, 특성 추가 등으로 실습 확장 가능 