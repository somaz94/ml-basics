"""
Visualization module for English sentiment analysis
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
    """Sentiment analysis visualizer"""
    
    def __init__(self, config: Dict[str, Any] = VIZ_CONFIG):
        self.config = config
        self._setup_plotting()
    
    def _setup_plotting(self):
        """Setup matplotlib configuration"""
        plt.rcParams['font.family'] = self.config['font_family']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, Any]]):
        """Visualize model performance comparison"""
        plt.figure(figsize=self.config['figure_size'])
        
        # Accuracy comparison
        plt.subplot(2, 3, 1)
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        colors = ['skyblue', 'lightcoral', 'lightgreen'][:len(model_names)]
        
        plt.bar(model_names, accuracies, color=colors)
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison')
        plt.ylim(0, 1)
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Confusion matrices
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
            plt.title(f'{name} Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
    
    def plot_wordclouds(self, df: pd.DataFrame):
        """Generate word clouds for positive/negative sentiments"""
        plt.figure(figsize=(12, 5))
        
        # Positive sentiment word cloud
        plt.subplot(1, 2, 1)
        positive_text = ' '.join(df[df['sentiment'] == 1]['processed_text'])
        
        if positive_text.strip():
            wordcloud_positive = WordCloud(
                width=self.config['wordcloud_size'][0],
                height=self.config['wordcloud_size'][1],
                background_color='white',
                colormap='Greens',
                max_words=100
            ).generate(positive_text)
            
            plt.imshow(wordcloud_positive, interpolation='bilinear')
        plt.axis('off')
        plt.title('Positive Sentiment Word Cloud')
        
        # Negative sentiment word cloud
        plt.subplot(1, 2, 2)
        negative_text = ' '.join(df[df['sentiment'] == 0]['processed_text'])
        
        if negative_text.strip():
            wordcloud_negative = WordCloud(
                width=self.config['wordcloud_size'][0],
                height=self.config['wordcloud_size'][1],
                background_color='white',
                colormap='Reds',
                max_words=100
            ).generate(negative_text)
            
            plt.imshow(wordcloud_negative, interpolation='bilinear')
        plt.axis('off')
        plt.title('Negative Sentiment Word Cloud')
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, tfidf_extractor, df: pd.DataFrame, top_k: int = 10):
        """Visualize TF-IDF feature importance"""
        plt.figure(figsize=(12, 6))
        
        feature_names = tfidf_extractor.get_feature_names()
        all_tfidf = tfidf_extractor.transform(df['processed_text'])
        
        positive_indices = df[df['sentiment'] == 1].index
        negative_indices = df[df['sentiment'] == 0].index
        
        positive_scores = all_tfidf[positive_indices].mean(axis=0).A1
        negative_scores = all_tfidf[negative_indices].mean(axis=0).A1
        
        top_positive_features = np.argsort(positive_scores)[-top_k:]
        top_negative_features = np.argsort(negative_scores)[-top_k:]
        
        # Positive features plot
        plt.subplot(1, 2, 1)
        top_positive_words = [feature_names[i] for i in top_positive_features]
        top_positive_values = [positive_scores[i] for i in top_positive_features]
        
        plt.barh(range(len(top_positive_words)), top_positive_values, color='green', alpha=0.7)
        plt.yticks(range(len(top_positive_words)), top_positive_words)
        plt.xlabel('Average TF-IDF Score')
        plt.title('Top Positive Sentiment Words')
        plt.gca().invert_yaxis()
        
        # Negative features plot
        plt.subplot(1, 2, 2)
        top_negative_words = [feature_names[i] for i in top_negative_features]
        top_negative_values = [negative_scores[i] for i in top_negative_features]
        
        plt.barh(range(len(top_negative_words)), top_negative_values, color='red', alpha=0.7)
        plt.yticks(range(len(top_negative_words)), top_negative_words)
        plt.xlabel('Average TF-IDF Score')
        plt.title('Top Negative Sentiment Words')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.show()
        
        return top_positive_words, top_negative_words