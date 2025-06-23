"""
한국어 감정 분석 시각화 모듈
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from typing import Dict, Any, List
from sklearn.metrics import confusion_matrix

from ..config import VIZ_CONFIG


class SentimentVisualizer:
    """감정 분석 결과 시각화 클래스"""
    
    def __init__(self, config: Dict[str, Any] = VIZ_CONFIG):
        self.config = config
        self._setup_plotting()
    
    def _setup_plotting(self):
        """matplotlib 설정"""
        plt.rcParams['font.family'] = self.config['font_family']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, Any]]):
        """모델 성능 비교 시각화"""
        plt.figure(figsize=self.config['figure_size'])
        
        # 정확도 비교
        plt.subplot(2, 3, 1)
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        colors = ['skyblue', 'lightcoral', 'lightgreen'][:len(model_names)]
        
        plt.bar(model_names, accuracies, color=colors)
        plt.ylabel('정확도')
        plt.title('모델 정확도 비교')
        plt.ylim(0, 1)
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # 혼동행렬
        for i, (name, result) in enumerate(results.items()):
            if 'confusion_matrix' in result:
                cm = result['confusion_matrix']
            else:
                y_test = result.get('y_test')
                y_pred = result.get('predictions')
                if y_test is not None and y_pred is not None:
                    cm = confusion_matrix(y_test, y_pred)
                else:
                    continue
            
            plt.subplot(2, 3, i + 2)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{name} 혼동행렬')
            plt.xlabel('예측값')
            plt.ylabel('실제값')
        
        plt.tight_layout()
        plt.show()
    
    def plot_wordclouds(self, df: pd.DataFrame):
        """긍정/부정 감정별 워드클라우드 생성"""
        plt.figure(figsize=(12, 5))
        
        # 긍정 감정 워드클라우드
        plt.subplot(1, 2, 1)
        positive_text = ' '.join(df[df['sentiment'] == 1]['processed_text'])
        
        if positive_text.strip():
            wordcloud_positive = WordCloud(
                font_path=self.config['font_path'],
                width=self.config['wordcloud_size'][0],
                height=self.config['wordcloud_size'][1],
                background_color='white',
                colormap='Greens',
                max_words=100
            ).generate(positive_text)
            
            plt.imshow(wordcloud_positive, interpolation='bilinear')
        plt.axis('off')
        plt.title('긍정 감정 워드클라우드')
        
        # 부정 감정 워드클라우드
        plt.subplot(1, 2, 2)
        negative_text = ' '.join(df[df['sentiment'] == 0]['processed_text'])
        
        if negative_text.strip():
            wordcloud_negative = WordCloud(
                font_path=self.config['font_path'],
                width=self.config['wordcloud_size'][0],
                height=self.config['wordcloud_size'][1],
                background_color='white',
                colormap='Reds',
                max_words=100
            ).generate(negative_text)
            
            plt.imshow(wordcloud_negative, interpolation='bilinear')
        plt.axis('off')
        plt.title('부정 감정 워드클라우드')
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, tfidf_extractor, df: pd.DataFrame, top_k: int = 10):
        """TF-IDF 특성 중요도 시각화"""
        plt.figure(figsize=(12, 6))
        
        feature_names = tfidf_extractor.get_feature_names()
        all_tfidf = tfidf_extractor.transform(df['processed_text'])
        
        positive_indices = df[df['sentiment'] == 1].index
        negative_indices = df[df['sentiment'] == 0].index
        
        positive_scores = all_tfidf[positive_indices].mean(axis=0).A1
        negative_scores = all_tfidf[negative_indices].mean(axis=0).A1
        
        top_positive_features = np.argsort(positive_scores)[-top_k:]
        top_negative_features = np.argsort(negative_scores)[-top_k:]
        
        # 긍정 특성 플롯
        plt.subplot(1, 2, 1)
        top_positive_words = [feature_names[i] for i in top_positive_features]
        top_positive_values = [positive_scores[i] for i in top_positive_features]
        
        plt.barh(range(len(top_positive_words)), top_positive_values, color='green', alpha=0.7)
        plt.yticks(range(len(top_positive_words)), top_positive_words)
        plt.xlabel('평균 TF-IDF 점수')
        plt.title('상위 긍정 감정 단어')
        plt.gca().invert_yaxis()
        
        # 부정 특성 플롯
        plt.subplot(1, 2, 2)
        top_negative_words = [feature_names[i] for i in top_negative_features]
        top_negative_values = [negative_scores[i] for i in top_negative_features]
        
        plt.barh(range(len(top_negative_words)), top_negative_values, color='red', alpha=0.7)
        plt.yticks(range(len(top_negative_words)), top_negative_words)
        plt.xlabel('평균 TF-IDF 점수')
        plt.title('상위 부정 감정 단어')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.show()
        
        return top_positive_words, top_negative_words
