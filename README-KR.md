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

---

각 폴더의 README.md에서 더 자세한 설명과 실행법을 확인할 수 있습니다.
