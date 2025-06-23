"""
한국어 감정 분석 전통적 머신러닝 모델
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
    """TF-IDF 특성 추출기"""
    
    def __init__(self, config: Dict[str, Any] = TFIDF_CONFIG):
        self.vectorizer = TfidfVectorizer(
            max_features=config['max_features'],
            ngram_range=config['ngram_range'],
            min_df=config['min_df'],
            max_df=config['max_df'],
            stop_words=config['stop_words']
        )
    
    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        """텍스트를 TF-IDF 특성으로 변환 (훈련용)"""
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts: pd.Series) -> np.ndarray:
        """텍스트를 TF-IDF 특성으로 변환 (테스트용)"""
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self) -> np.ndarray:
        """특성 이름들 반환"""
        return self.vectorizer.get_feature_names_out()


class TraditionalModels:
    """전통적인 머신러닝 모델들을 관리하는 클래스"""
    
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
        """모든 모델을 훈련하고 평가"""
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\n{name} 학습 중...")
            
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
            
            print(f"{name} 정확도: {acc:.3f}")
        
        return self.results
    
    def predict_single(self, model_name: str, features) -> int:
        """단일 샘플에 대한 예측"""
        if model_name not in self.results:
            raise ValueError(f"모델 '{model_name}'이 훈련되지 않았습니다.")
        
        model = self.results[model_name]['model']
        return model.predict(features)[0]
