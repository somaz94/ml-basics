# ml-basics

머신러닝 입문자를 위한 대표 실습 예제 모음입니다. 각 폴더별로 다양한 데이터와 알고리즘을 활용한 실습 코드를 제공합니다.

<br/>

## 폴더별 실습 주제

### 1. intro
- **주제:** 과일 데이터로 배우는 지도학습, 비지도학습, 유사도 비교
- **예제:** DecisionTree, KMeans, Cosine Similarity
- **실행법:**


```bash
cd intro
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 fruit_ml_examples_kr.py
```

<br/>

### 2. iris_classification
- **주제:** 붓꽃(Iris) 데이터셋 분류/군집화 실습
- **예제:** SVM, KNN, LogisticRegression, KMeans, 2D/3D 시각화
- **실행법:**


```bash
cd iris_classification
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 iris_classification_kr.py
```

<br/>

### 3. wine_quality
- **주제:** 와인 품질 데이터 분류/회귀/군집화 실습
- **예제:** RandomForest, LinearRegression, KMeans, 산점도 시각화
- **실행법:**


```bash
cd wine_quality
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 wine_quality_kr.py
  ```

<br/>

### 4. titanic_survival
- **주제:** 타이타닉 생존자 예측 분류 실습
- **예제:** RandomForest, LogisticRegression, 데이터 전처리, 혼동행렬
- **실행법:**


```bash
cd titanic_survival
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 titanic_surviva_kr_.py
```

<br/>

### 5. boston_housing
- **주제:** 보스턴 집값 예측 회귀/군집화 실습
- **예제:** LinearRegression, RandomForest, KMeans, 산점도 시각화
- **실행법:**

```bash
cd boston_housing
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 boston_housing_kr.py
```


<br/>

### 6. mnist_digit_classification
- **주제:** MNIST 손글씨 숫자 분류/딥러닝 실습
- **예제:** CNN, RandomForest, SVM, KMeans, 이미지 시각화
- **실행법:**


```bash
cd mnist_digit_classification
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 mnist_digit_classification_kr.py
```

<br/>

### 7. breast_cancer_diagnosis
- **주제:** 유방암 진단 의료 데이터 분류/군집화 실습
- **예제:** LogisticRegression, RandomForest, 신경망, KMeans, 의료 AI
- **실행법:**


```bash
cd breast_cancer_diagnosis
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 breast_cancer_diagnosis_kr.py
```

<br/>

### 8. credit_card_fraud_detection
- **주제:** 신용카드 사기 탐지 불균형 데이터 분류 실습
- **예제:** RandomForest, SMOTE, 신경망, KMeans, 비즈니스 AI
- **실행법:**


```bash
cd credit_card_fraud_detection
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 credit_card_fraud_detection_kr.py
```

<br/>

### 9. sentiment_analysis
- **주제:** 감정 분석 NLP 텍스트 분류 실습
- **예제:** TF-IDF, NaiveBayes, SVM, 신경망, 워드클라우드
- **실행법:**


```bash
cd sentiment_analysis
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 sentiment_analysis_kr.py
```

<br/>

### 10. stock_price_prediction
- **주제:** 주식/지수 시계열 데이터 회귀 실습 (미국/코스피)
- **예제:** LinearRegression, LSTM, 시계열 시각화, 예측 결과 표 출력
- **실행법:**

#### (1) 미국 주식(Yahoo Finance)
```bash
cd stock_price_prediction
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 stock_price_prediction_kr.py
```

#### (2) 코스피 지수(네이버 금융)
```bash
cd stock_price_prediction
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 stock_price_prediction_kr_naver.py
```

### 11. customer_churn_prediction
- **주제:** 고객 이탈 예측 (고객 생애가치 분석 실습)
- **예제:** RandomForest, LogisticRegression, 특성 중요도
- **실행법:**


```bash
cd customer_churn_prediction
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 customer_churn_prediction_kr.py
```

<br/>

### 12. image_classification_cifar10
- **주제:** CIFAR-10 데이터셋 이미지 분류 실습
- **예제:** CNN, 데이터 증강, PyTorch
- **실행법:**


```bash
cd image_classification_cifar10
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 image_classification_cifar10_kr.py
```

---

각 폴더의 README.md에서 더 자세한 설명과 실행법을 확인할 수 있습니다.
