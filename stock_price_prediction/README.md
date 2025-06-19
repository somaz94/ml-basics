# Stock Price Prediction (Practice 10)

- **Topic:** Time Series Regression Practice
- **Examples:** LinearRegression, LSTM, Time Series Visualization
- **Features:** Basic time series analysis, real stock data (US and Korea)

---

## Practice Structure

1. **US Stock (Yahoo Finance) Practice**
   - Time series regression using global stock data (e.g., Apple)
   - Data source: Yahoo Finance
   - File: `stock_price_prediction.py`

2. **KOSPI Index (Naver Finance) Practice**
   - Crawling daily KOSPI index data from Naver Finance
   - Time series analysis and prediction for Korean market
   - File: `stock_price_prediction_kr_naver.py`

---

## How to Run

### 1. US Stock (Yahoo Finance)
```bash
cd stock_price_prediction
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 stock_price_prediction.py
```

### 2. KOSPI (Naver Finance)
```bash
cd stock_price_prediction
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 stock_price_prediction_kr_naver.py
```

---

## Main Contents & Practical Tips
- **Real Data:** Practice with both global (US) and domestic (KOSPI) time series data
- **Model Comparison:** Compare LinearRegression and LSTM deep learning models
- **Time Series Analysis:** Windowing, feature creation, visualization, overfitting caution, etc.
- **Crawling Practice:** Naver Finance data is collected via web crawling; handle missing/outlier values
- **Extensibility:** Easily expand to other stocks, indices, window sizes, or features 