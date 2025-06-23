"""
Traditional machine learning models for English sentiment analysis
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

from ..config import TFIDF_CONFIG, MODEL_CONFIG


class TfidfFeatureExtractor:
    """TF-IDF feature extractor"""
    
    def __init__(self, config: Dict[str, Any] = TFIDF_CONFIG):
        self.vectorizer = TfidfVectorizer(
            max_features=config['max_features'],
            ngram_range=config['ngram_range'],
            min_df=config['min_df'],
            max_df=config['max_df'],
            stop_words=config['stop_words']
        )
    
    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        """Fit and transform texts to TF-IDF features"""
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts: pd.Series) -> np.ndarray:
        """Transform texts to TF-IDF features"""
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self) -> np.ndarray:
        """Get feature names"""
        return self.vectorizer.get_feature_names_out()


class TraditionalModels:
    """Traditional machine learning models manager"""
    
    def __init__(self):
        self.models = {
            'NaiveBayes': MultinomialNB(),
            'SVM': SVC(
                kernel=MODEL_CONFIG['svm_kernel'], 
                random_state=MODEL_CONFIG['random_state']
            )
        }
        self.results = {}
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test) -> Dict[str, Dict[str, Any]]:
        """Train and evaluate all models"""
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            self.results[name] = {
                'accuracy': acc,
                'predictions': y_pred,
                'confusion_matrix': cm,
                'classification_report': report,
                'model': model
            }
            
            print(f"{name} Accuracy: {acc:.3f}")
        
        return self.results
    
    def predict_single(self, model_name: str, features) -> int:
        """Predict single sample"""
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not trained.")
        
        model = self.results[model_name]['model']
        return model.predict(features)[0]