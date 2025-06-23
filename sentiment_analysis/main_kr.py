"""
메인 애플리케이션 - 한국어 감정 분석
리팩토링된 모듈형 구조
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src_kr.data import KoreanSentimentDataGenerator, TextPreprocessor, create_train_test_split
from src_kr.models import TfidfFeatureExtractor, TraditionalModels, NeuralNetworkTrainer
from src_kr.visualization import SentimentVisualizer
from src_kr.utils import (
    setup_nltk, 
    print_model_summary, 
    print_nlp_insights, 
    print_best_practices, 
    print_top_features,
    ModelPredictor
)


class KoreanSentimentAnalysisApp:
    """한국어 감정 분석 메인 애플리케이션"""
    
    def __init__(self):
        self.data_generator = KoreanSentimentDataGenerator()
        self.text_preprocessor = TextPreprocessor()
        self.tfidf_extractor = TfidfFeatureExtractor()
        self.traditional_models = TraditionalModels()
        self.neural_network = None
        self.visualizer = SentimentVisualizer()
        self.predictor = None
        
        # 데이터 관련 변수
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_tfidf = None
        self.X_test_tfidf = None
        
        # 결과 관련 변수
        self.results = {}
    
    def setup_environment(self):
        """환경 설정"""
        print("📘 환경 설정 중...")
        setup_nltk()
        print("✅ 환경 설정 완료")
    
    def generate_and_preprocess_data(self):
        """데이터 생성 및 전처리"""
        print("\n📘 데이터 생성 및 전처리")
        
        print("합성 감정 분석 데이터셋 생성 중...")
        self.df = self.data_generator.generate_dataset()
        print(f"데이터셋 형태: {self.df.shape}")
        print(f"감정 분포: {self.df['sentiment'].value_counts().to_dict()}")
        
        print("텍스트 전처리 중...")
        self.df = self.text_preprocessor.process_dataframe(self.df)
        
        self.X_train, self.X_test, self.y_train, self.y_test = create_train_test_split(self.df)
        print(f"훈련 세트: {len(self.X_train)}, 테스트 세트: {len(self.X_test)}")
        
        print("✅ 데이터 준비 완료")
    
    def extract_features(self):
        """TF-IDF 특성 추출"""
        print("\n📘 TF-IDF 특성 추출")
        
        self.X_train_tfidf = self.tfidf_extractor.fit_transform(self.X_train)
        self.X_test_tfidf = self.tfidf_extractor.transform(self.X_test)
        
        print(f"TF-IDF 특성 수: {self.X_train_tfidf.shape[1]}")
        print("✅ 특성 추출 완료")
    
    def train_traditional_models(self):
        """전통적인 머신러닝 모델 훈련"""
        print("\n📘 전통적 머신러닝 모델 훈련")
        
        traditional_results = self.traditional_models.train_and_evaluate(
            self.X_train_tfidf, self.X_test_tfidf, self.y_train, self.y_test
        )
        
        self.results.update(traditional_results)
        print("✅ 전통적 모델 훈련 완료")
    
    def train_neural_network(self):
        """신경망 훈련"""
        print("\n📘 신경망 훈련")
        
        input_size = self.X_train_tfidf.shape[1]
        self.neural_network = NeuralNetworkTrainer(input_size)
        
        nn_result = self.neural_network.train(
            self.X_train_tfidf, self.X_test_tfidf, self.y_train, self.y_test
        )
        
        self.results['NeuralNetwork'] = nn_result
        print("✅ 신경망 훈련 완료")
    
    def visualize_results(self):
        """결과 시각화"""
        print("\n📘 결과 시각화")
        
        # 모델 성능 비교
        viz_results = {}
        for name, result in self.results.items():
            viz_results[name] = {
                'accuracy': result['accuracy'],
                'predictions': result['predictions'],
                'y_test': self.y_test
            }
            if 'confusion_matrix' in result:
                viz_results[name]['confusion_matrix'] = result['confusion_matrix']
        
        self.visualizer.plot_model_comparison(viz_results)
        self.visualizer.plot_wordclouds(self.df)
        
        top_positive, top_negative = self.visualizer.plot_feature_importance(
            self.tfidf_extractor, self.df
        )
        
        print("✅ 시각화 완료")
        return top_positive, top_negative
    
    def analyze_results(self, top_positive_words, top_negative_words):
        """결과 분석 및 출력"""
        print("\n📘 결과 분석")
        
        print_model_summary(self.results)
        print_nlp_insights()
        print_top_features(top_positive_words, top_negative_words)
        print_best_practices()
        
        print("✅ 결과 분석 완료")
    
    def setup_predictor(self):
        """예측기 설정"""
        self.predictor = ModelPredictor(
            self.traditional_models,
            self.neural_network,
            self.tfidf_extractor,
            self.text_preprocessor
        )
    
    def demo_predictions(self):
        """샘플 예측 데모"""
        print("\n📘 샘플 예측 데모")
        
        sample_texts = [
            "이 제품은 정말 환상적입니다!",
            "끔찍한 품질, 매우 실망했습니다.",
            "좋지만 더 나을 수 있어요.",
            "놀라운 경험, 강력히 추천합니다!"
        ]
        
        for text in sample_texts:
            result = self.predictor.predict_sentiment(text)
            print(f"텍스트: '{result['text']}'")
            print(f"감정: {result['sentiment']}")
            print(f"모델 예측: {result['predictions']}")
            print("-" * 40)
        
        print("✅ 예측 데모 완료")
    
    def run(self):
        """전체 파이프라인 실행"""
        print("🚀 한국어 감정 분석 애플리케이션 시작")
        print("=" * 60)
        
        try:
            self.setup_environment()
            self.generate_and_preprocess_data()
            self.extract_features()
            self.train_traditional_models()
            self.train_neural_network()
            top_positive, top_negative = self.visualize_results()
            self.analyze_results(top_positive, top_negative)
            self.setup_predictor()
            self.demo_predictions()
            
            print("\n🎉 한국어 감정 분석 애플리케이션 완료!")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()


def main():
    """메인 함수"""
    app = KoreanSentimentAnalysisApp()
    app.run()


if __name__ == "__main__":
    main()
