"""
Utility functions for English sentiment analysis
"""

import nltk
from typing import Dict, Any, List


def setup_nltk():
    """Download required NLTK data"""
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
    """Print model performance summary"""
    print("\n📘 Model Comparison and NLP Insights")
    print("=" * 50)
    print("Model Performance Summary:")
    print("-" * 30)
    for name, result in results.items():
        print(f"{name:15} | Accuracy: {result['accuracy']:.3f}")
    print("=" * 50)


def print_nlp_insights():
    """Print NLP insights"""
    print("\n📘 NLP Interpretation:")
    print("- TF-IDF: Term frequency-inverse document frequency for feature extraction")
    print("- Naive Bayes: Probabilistic classifier based on Bayes theorem")
    print("- SVM: Support Vector Machine for text classification")
    print("- Neural Network: Deep learning approach for sentiment analysis")


def print_best_practices():
    """Print NLP best practices"""
    print("\n📘 NLP Best Practices:")
    print("- Text preprocessing: Lowercase, remove special characters")
    print("- TF-IDF vectorization: Capture word importance")
    print("- Stop words removal: Focus on meaningful words")
    print("- N-gram features: Capture word combinations")
    print("- Cross-validation: Ensure model generalization")


def print_top_features(top_positive_words: List[str], top_negative_words: List[str], top_k: int = 5):
    """Print top features"""
    print(f"\n📘 Key Features (Top {top_k} Positive Words):")
    for i, word in enumerate(top_positive_words[-top_k:], 1):
        print(f"{i}. {word}")

    print(f"\n📘 Key Features (Top {top_k} Negative Words):")
    for i, word in enumerate(top_negative_words[-top_k:], 1):
        print(f"{i}. {word}")


class ModelPredictor:
    """Unified prediction class"""
    
    def __init__(self, traditional_models, neural_network, tfidf_extractor, text_preprocessor):
        self.traditional_models = traditional_models
        self.neural_network = neural_network
        self.tfidf_extractor = tfidf_extractor
        self.text_preprocessor = text_preprocessor
    
    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """Predict sentiment for text"""
        processed_text = self.text_preprocessor.preprocess_text(text)
        tfidf_features = self.tfidf_extractor.transform([processed_text])
        
        predictions = {}
        
        # Traditional models
        for name in self.traditional_models.results.keys():
            predictions[name] = self.traditional_models.predict_single(name, tfidf_features)
        
        # Neural network
        if self.neural_network:
            predictions['NeuralNetwork'] = self.neural_network.predict_single(tfidf_features)
        
        # Majority vote
        prediction_values = list(predictions.values())
        majority_vote = 1 if sum(prediction_values) > len(prediction_values) / 2 else 0
        sentiment_label = "Positive" if majority_vote == 1 else "Negative"
        
        return {
            'text': text,
            'processed_text': processed_text,
            'predictions': predictions,
            'majority_vote': majority_vote,
            'sentiment': sentiment_label
        }