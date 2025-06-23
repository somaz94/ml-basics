"""
Korean sentiment analysis를 위한 데이터 생성 및 전처리
"""

import numpy as np
import pandas as pd
import re
from typing import List, Tuple
from sklearn.model_selection import train_test_split

from ..config import DATA_CONFIG


class KoreanSentimentDataGenerator:
    """한국어 감정 분석 데이터 생성기"""
    
    def __init__(self, sample_size: int = DATA_CONFIG['sample_size']):
        self.sample_size = sample_size
        self.positive_reviews = [
            "이 제품은 정말 훌륭합니다! 너무 만족스러워요.",
            "품질이 우수하고 서비스도 좋습니다.",
            "환상적인 경험이었어요, 강력히 추천합니다!",
            "기대를 뛰어넘는 멋진 제품입니다.",
            "가격 대비 품질이 훌륭하고 매우 만족합니다.",
            "성능과 신뢰성이 뛰어납니다.",
            "최고의 품질, 지금까지 최고의 구매였어요!",
            "놀라운 기능들, 정말 마음에 듭니다.",
            "제 필요에 완벽하고, 훌륭한 선택입니다.",
            "디자인과 기능성이 뛰어납니다."
        ]
        
        self.negative_reviews = [
            "끔찍한 제품, 완전히 돈 낭비입니다.",
            "품질이 형편없고 매우 실망스럽습니다.",
            "최악의 경험, 추천하지 않습니다.",
            "서비스도 나쁘고 제품도 형편없어요.",
            "지금까지 최악의 구매, 피하세요.",
            "실망스러운 품질, 가치가 없습니다.",
            "디자인도 나쁘고 기능성도 떨어집니다.",
            "끔찍한 고객 서비스 경험입니다.",
            "쓸모없는 제품, 구매를 후회합니다.",
            "성능이 나쁘고 신뢰성에 문제가 있습니다."
        ]
    
    def _generate_variations(self, base_reviews: List[str], count: int) -> List[str]:
        """기본 리뷰에서 다양한 변형 생성"""
        reviews = []
        for _ in range(count):
            base = np.random.choice(base_reviews)
            variations = [
                f"정말로 {base}",
                f"{base} 확실히!",
                f"제 생각에는 {base}",
                f"{base} 제 의견입니다.",
                f"절대적으로 {base}",
                f"{base} 의심의 여지가 없어요.",
                f"명백히 {base}",
                f"{base} 의문의 여지가 없습니다.",
                f"분명히 {base}",
                f"{base} 확실합니다."
            ]
            reviews.append(np.random.choice(variations))
        return reviews
    
    def generate_dataset(self) -> pd.DataFrame:
        """합성 감정 분석 데이터셋 생성"""
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
    """한국어 텍스트 전처리 클래스"""
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """한국어 텍스트 전처리 함수"""
        text = text.lower()
        # 한글, 영문, 공백만 유지
        text = re.sub(r'[^가-힣a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame의 텍스트 컬럼 처리"""
        df = df.copy()
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        return df


def create_train_test_split(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """훈련/테스트 데이터 분할"""
    return train_test_split(
        df['processed_text'], 
        df['sentiment'],
        test_size=DATA_CONFIG['test_size'],
        random_state=DATA_CONFIG['random_state'],
        stratify=df['sentiment']
    )
