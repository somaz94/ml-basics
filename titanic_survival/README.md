# Practice 4: Titanic Survivor Prediction/Clustering Practice

This directory provides practice examples for supervised learning (survival prediction), unsupervised learning (clustering), and visualization using the Titanic dataset.

<br/>

## Example Files
- `titanic_survival.py`: Survival prediction using DecisionTree, RandomForest, LogisticRegression, passenger clustering using KMeans, scatter plot visualization

<br/>

## Execution Method

1. Create and activate virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. Install dependencies
```bash
pip3 install -r requirements.txt
```

3. Run examples
```bash
python3 titanic_survival.py
```

<br/>


## Main Contents
- **Supervised Learning**: Survival prediction using DecisionTree, RandomForest, LogisticRegression and accuracy comparison
- **Unsupervised Learning**: Passenger data clustering using KMeans, comparison with actual survivors
- **Visualization**: Age-fare-survival scatter plot, clustering result visualization

<br/>

## Notes
- You can extend with various feature engineering, missing value handling, and visualization methods. 