# Practice 7: Breast Cancer Diagnosis

<br/>

## Overview
- Breast Cancer Wisconsin dataset for medical AI classification and clustering practice.
- Use scikit-learn's load_breast_cancer to load the dataset and apply various algorithms including LogisticRegression, RandomForest, Neural Network, and KMeans.

<br/>

## Dataset Introduction
- **Breast Cancer Wisconsin**: 569 samples with 30 features for breast cancer diagnosis
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- Features: 30 numerical features computed from cell nucleus characteristics
- Target: Binary classification (0: Benign, 1: Malignant)

<br/>

## Environment Setup and Execution
```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install packages
pip install -r requirements.txt

# 3. Run practice code
python breast_cancer_diagnosis.py
```

<br/>

## Main Practice Contents
- Data loading and preprocessing (StandardScaler, train_test_split)
- Supervised Learning: LogisticRegression, RandomForest for cancer classification
- Deep Learning: Neural Network with PyTorch for medical diagnosis
- Unsupervised Learning: KMeans clustering with 2 clusters
- Visualization: Feature importance, confusion matrices, ROC curves, PCA clustering
- Medical interpretation and clinical recommendations

<br/>

## Result Interpretation
- Supervised Learning: Compare accuracy and AUC between LogisticRegression, RandomForest, and Neural Network
- Deep Learning: Neural Network typically achieves high accuracy for medical diagnosis
- Unsupervised Learning: KMeans groups similar cancer patterns without labels
- Medical Insights: Feature importance shows key diagnostic characteristics

<br/>

## Key Learning Points
- **Medical AI**: Understanding the importance of accuracy in healthcare applications
- **Feature Engineering**: Identifying important diagnostic features
- **Model Evaluation**: Using multiple metrics (accuracy, AUC, confusion matrix)
- **Clinical Applications**: Real-world medical diagnosis scenarios

<br/>

## Medical Context
- **Benign (0)**: Non-cancerous tumors, generally not life-threatening
- **Malignant (1)**: Cancerous tumors requiring immediate treatment
- **False Negatives**: More dangerous than false positives in medical diagnosis
- **Feature Importance**: Understanding which cell characteristics are most diagnostic

<br/>

## Conclusion
- Practice with real medical data for cancer diagnosis
- Compare traditional machine learning and deep learning approaches
- Understand the critical importance of accuracy in medical AI
- Learn feature importance analysis for clinical decision making

<br/>

## Reference
- [UCI Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [Scikit-learn Breast Cancer](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)
- [Medical AI Best Practices](https://www.nature.com/articles/s41591-019-0648-5)
- [ROC Curve Analysis](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) 