"""
Utilities module for English sentiment analysis
"""

from .helpers import (
    setup_nltk, 
    print_model_summary, 
    print_nlp_insights, 
    print_best_practices, 
    print_top_features,
    ModelPredictor
)

__all__ = [
    'setup_nltk', 
    'print_model_summary', 
    'print_nlp_insights', 
    'print_best_practices', 
    'print_top_features',
    'ModelPredictor'
]
