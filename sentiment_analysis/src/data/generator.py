"""
Data generation and preprocessing for English sentiment analysis
"""

import numpy as np
import pandas as pd
import re
from typing import List, Tuple
from sklearn.model_selection import train_test_split

from ..config import DATA_CONFIG


class EnglishSentimentDataGenerator:
    """English sentiment analysis data generator"""
    
    def __init__(self, sample_size: int = DATA_CONFIG['sample_size']):
        self.sample_size = sample_size
        self.positive_reviews = [
            "This product is amazing! I love it so much.",
            "Excellent quality and great service.",
            "Wonderful experience, highly recommended!",
            "Fantastic product, exceeded my expectations.",
            "Great value for money, very satisfied.",
            "Outstanding performance and reliability.",
            "Superb quality, best purchase ever!",
            "Incredible features, absolutely love it.",
            "Perfect for my needs, excellent choice.",
            "Brilliant design and functionality."
        ]
        
        self.negative_reviews = [
            "Terrible product, complete waste of money.",
            "Awful quality, very disappointed.",
            "Horrible experience, would not recommend.",
            "Poor service and bad product.",
            "Worst purchase ever, avoid this.",
            "Disappointing quality, not worth it.",
            "Bad design and poor functionality.",
            "Terrible customer service experience.",
            "Useless product, regret buying it.",
            "Poor performance and reliability issues."
        ]
    
    def _generate_variations(self, base_reviews: List[str], count: int) -> List[str]:
        """Generate varied reviews from base reviews"""
        reviews = []
        for _ in range(count):
            base = np.random.choice(base_reviews)
            variations = [
                f"Really {base}",
                f"{base} Definitely!",
                f"I think {base}",
                f"{base} In my opinion.",
                f"Absolutely {base}",
                f"{base} No doubt about it.",
                f"Clearly {base}",
                f"{base} Without question.",
                f"Obviously {base}",
                f"{base} For sure."
            ]
            reviews.append(np.random.choice(variations))
        return reviews
    
    def generate_dataset(self) -> pd.DataFrame:
        """Generate synthetic sentiment analysis dataset"""
        np.random.seed(DATA_CONFIG['random_state'])
        
        positive_samples = self._generate_variations(self.positive_reviews, self.sample_size)
        negative_samples = self._generate_variations(self.negative_reviews, self.sample_size)
        
        texts = positive_samples + negative_samples
        labels = [1] * self.sample_size + [0] * self.sample_size
        
        return pd.DataFrame({
            'text': texts,
            'sentiment': labels
        })


class TextPreprocessor:
    """English text preprocessing class"""
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """English text preprocessing function"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process text column in DataFrame"""
        df = df.copy()
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        return df


def create_train_test_split(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create train/test data split"""
    return train_test_split(
        df['processed_text'], 
        df['sentiment'],
        test_size=DATA_CONFIG['test_size'],
        random_state=DATA_CONFIG['random_state'],
        stratify=df['sentiment']
    )