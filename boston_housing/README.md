# Practice 5: Boston Housing Price Prediction (Boston Housing)

<br/>

## Overview
- Practice for supervised learning (regression), unsupervised learning (clustering), and visualization using the Boston housing price dataset.
- Load dataset using scikit-learn's fetch_openml and apply various algorithms including LinearRegression, RandomForest, and KMeans.

<br/>

## Dataset Introduction
- **Boston Housing**: Housing price data from 1970s Boston area with 13 features (number of rooms, crime rate, etc.)
- Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Housing)

<br/>

## Environment Setup and Execution
```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install packages
pip3 install -r requirements.txt

# 3. Run practice code
python3 boston_housing.py
```

<br/>

## Main Practice Contents
- Data loading and preprocessing (fetch_openml, pandas)
- Supervised Learning: House price prediction using LinearRegression, RandomForestRegressor, MSE output
- Unsupervised Learning: Regional clustering using KMeans, sample count output by cluster
- Visualization: Room count (RM) - house price (MEDV) scatter plot, clustering result visualization
- Korean font setting (AppleGothic)

<br/>

## Result Interpretation
- Supervised Learning: Regression performance comparison using MSE (Mean Squared Error)
- Unsupervised Learning: 3-cluster classification using KMeans, possible to check characteristics by cluster
- Visualization: Confirmation of increasing house price trend with increasing room count

<br/>

## Conclusion
- You can visually understand the relationship between features and targets, and differences in characteristics by cluster through practice with various regression/clustering algorithms.

<br/>

## Reference
- [scikit-learn Boston Housing](https://scikit-learn.org/stable/datasets/toy_dataset.html#boston-house-prices-dataset)
- [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Housing)
- [Kaggle Boston Housing](https://www.kaggle.com/datasets/altavish/boston-housing-dataset) 