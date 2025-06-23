"""
한국어 감정 분석 유틸리티 함수들
"""

import nltk
from typing import Dict, Any, List


def setup_nltk():
    """NLTK 필요 데이터 다운로드"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')


def print_model_summary(results: Dict[str, Dict[str, Any]]):
    """모델 성능 요약 출력"""
    print("\n📘 모델 비교 및 NLP 통찰")
    print("=" * 50)
    print("모델 성능 요약:")
    print("-" * 30)
    for name, result in results.items():
        print(f"{name:15} | 정확도: {result['accuracy']:.3f}")
    print("=" * 50)


def print_nlp_insights():
    """NLP 관련 해석 출력"""
    print("\n📘 NLP 해석:")
    print("- TF-IDF: 특성 추출을 위한 용어 빈도-역문서 빈도")
    print("- Naive Bayes: 베이즈 정리에 기반한 확률적 분류기")
    print("- SVM: 텍스트 분류를 위한 서포트 벡터 머신")
    print("- 신경망: 감정 분석을 위한 딥러닝 접근법")


def print_best_practices():
    """NLP 모범 사례 출력"""
    print("\n📘 NLP 모범 사례:")
    print("- 텍스트 전처리: 소문자 변환, 특수문자 제거")
    print("- TF-IDF 벡터화: 단어 중요도 포착")
    print("- 불용어 제거: 의미 있는 단어에 집중")
    print("- N-gram 특성: 단어 조합 포착")
    print("- 교차 검증: 모델 일반화 보장")


def print_top_features(top_positive_words: List[str], top_negative_words: List[str], top_k: int = 5):
    """상위 특성 출력"""
    print(f"\n📘 주요 특성 (상위 {top_k}개 긍정 단어):")
    for i, word in enumerate(top_positive_words[-top_k:], 1):
        print(f"{i}. {word}")

    print(f"\n📘 주요 특성 (상위 {top_k}개 부정 단어):")
    for i, word in enumerate(top_negative_words[-top_k:], 1):
        print(f"{i}. {word}")


class ModelPredictor:
    """통합 예측 클래스"""
    
    def __init__(self, traditional_models, neural_network, tfidf_extractor, text_preprocessor):
        self.traditional_models = traditional_models
        self.neural_network = neural_network
        self.tfidf_extractor = tfidf_extractor
        self.text_preprocessor = text_preprocessor
    
    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """텍스트의 감정을 예측"""
        processed_text = self.text_preprocessor.preprocess_text(text)
        tfidf_features = self.tfidf_extractor.transform([processed_text])
        
        predictions = {}
        
        # 전통적 모델들
        for name in self.traditional_models.results.keys():
            predictions[name] = self.traditional_models.predict_single(name, tfidf_features)
        
        # 신경망
        if self.neural_network:
            predictions['NeuralNetwork'] = self.neural_network.predict_single(tfidf_features)
        
        # 다수결 투표
        prediction_values = list(predictions.values())
        majority_vote = 1 if sum(prediction_values) > len(prediction_values) / 2 else 0
        sentiment_label = "긍정" if majority_vote == 1 else "부정"
        
        return {
            'text': text,
            'processed_text': processed_text,
            'predictions': predictions,
            'majority_vote': majority_vote,
            'sentiment': sentiment_label
        }
