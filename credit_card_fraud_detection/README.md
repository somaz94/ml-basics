# Practice 8: Credit Card Fraud Detection

<br/>

## Overview
- Synthetic credit card fraud dataset for imbalanced data classification practice.
- Use various techniques including SMOTE, RandomForest, LogisticRegression, and Neural Network to handle class imbalance and detect fraudulent transactions.

<br/>

## Dataset Introduction
- **Synthetic Credit Card Fraud**: 10,000 transactions with 29 features (28 anonymized + amount)
- Features: V1-V28 (anonymized transaction features) + Amount
- Target: Binary classification (0: Normal, 1: Fraud)
- Class Imbalance: 99.5% normal transactions, 0.5% fraudulent transactions

<br/>

## Environment Setup and Execution
```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install packages
pip3 install -r requirements.txt

# 3. Run practice code
python3 credit_card_fraud_detection.py
```

<br/>

## Main Practice Contents
- Synthetic dataset generation with realistic fraud patterns
- Data preprocessing and class imbalance analysis
- Baseline models on imbalanced data (RandomForest, LogisticRegression)
- SMOTE oversampling for balanced training data
- Deep Learning: Neural Network with PyTorch
- Visualization: Class distribution, confusion matrices, ROC curves, precision-recall curves
- Business insights and cost analysis

<br/>

## Result Interpretation
- Baseline Models: Compare performance on imbalanced data
- SMOTE Balanced Models: Improved fraud detection with balanced classes
- Deep Learning: Neural Network performance on balanced data
- Business Impact: False negatives vs false positives cost analysis

<br/>

## Key Learning Points
- **Imbalanced Data**: Understanding the challenges of skewed class distributions
- **SMOTE**: Synthetic Minority Over-sampling Technique for balancing data
- **Fraud Detection**: Real-world business application with critical accuracy requirements
- **Cost Analysis**: Different costs for false positives vs false negatives

<br/>

## Business Context
- **Normal (0)**: Legitimate transactions - approve
- **Fraud (1)**: Fraudulent transactions - decline
- **False Positive**: Legitimate transaction flagged as fraud (customer inconvenience)
- **False Negative**: Fraudulent transaction missed (financial loss)

<br/>

## Technical Features
- **SMOTE**: Generates synthetic samples for minority class
- **ROC Curves**: Evaluate model discrimination ability
- **Precision-Recall Curves**: Important for imbalanced datasets
- **Feature Importance**: Identify key fraud indicators

<br/>

## Conclusion
- Practice with realistic imbalanced dataset for fraud detection
- Compare baseline vs balanced model performance
- Understand the importance of proper evaluation metrics for imbalanced data
- Learn business implications of fraud detection systems

<br/>

## Reference
- [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)
- [Credit Card Fraud Detection Best Practices](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Handling Imbalanced Data](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)
- [ROC Curve Analysis](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) 