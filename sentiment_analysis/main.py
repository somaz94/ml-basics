"""
Main application for English Sentiment Analysis
Refactored modular structure
"""

import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data import EnglishSentimentDataGenerator, TextPreprocessor, create_train_test_split
from src.models import TfidfFeatureExtractor, TraditionalModels, NeuralNetworkTrainer
from src.visualization import SentimentVisualizer
from src.utils import (
    setup_nltk, 
    print_model_summary, 
    print_nlp_insights, 
    print_best_practices, 
    print_top_features,
    ModelPredictor
)


class EnglishSentimentAnalysisApp:
    """English sentiment analysis main application"""
    
    def __init__(self):
        self.data_generator = EnglishSentimentDataGenerator()
        self.text_preprocessor = TextPreprocessor()
        self.tfidf_extractor = TfidfFeatureExtractor()
        self.traditional_models = TraditionalModels()
        self.neural_network = None
        self.visualizer = SentimentVisualizer()
        self.predictor = None
        
        # Data variables
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_tfidf = None
        self.X_test_tfidf = None
        
        # Results
        self.results = {}
    
    def setup_environment(self):
        """Setup environment"""
        print("📘 Setting up environment...")
        setup_nltk()
        print("✅ Environment setup complete")
    
    def generate_and_preprocess_data(self):
        """Generate and preprocess data"""
        print("\n📘 Data Generation and Preprocessing")
        
        print("Generating synthetic sentiment analysis dataset...")
        self.df = self.data_generator.generate_dataset()
        print(f"Dataset shape: {self.df.shape}")
        print(f"Sentiment distribution: {self.df['sentiment'].value_counts().to_dict()}")
        
        print("Preprocessing text...")
        self.df = self.text_preprocessor.process_dataframe(self.df)
        
        self.X_train, self.X_test, self.y_train, self.y_test = create_train_test_split(self.df)
        print(f"Training set: {len(self.X_train)}, Test set: {len(self.X_test)}")
        
        print("✅ Data preparation complete")
    
    def extract_features(self):
        """Extract TF-IDF features"""
        print("\n📘 TF-IDF Feature Extraction")
        
        self.X_train_tfidf = self.tfidf_extractor.fit_transform(self.X_train)
        self.X_test_tfidf = self.tfidf_extractor.transform(self.X_test)
        
        print(f"TF-IDF features: {self.X_train_tfidf.shape[1]}")
        print("✅ Feature extraction complete")
    
    def train_traditional_models(self):
        """Train traditional machine learning models"""
        print("\n📘 Traditional Machine Learning Models")
        
        traditional_results = self.traditional_models.train_and_evaluate(
            self.X_train_tfidf, self.X_test_tfidf, self.y_train, self.y_test
        )
        
        self.results.update(traditional_results)
        print("✅ Traditional models training complete")
    
    def train_neural_network(self):
        """Train neural network"""
        print("\n📘 Neural Network Training")
        
        input_size = self.X_train_tfidf.shape[1]
        self.neural_network = NeuralNetworkTrainer(input_size)
        
        nn_result = self.neural_network.train(
            self.X_train_tfidf, self.X_test_tfidf, self.y_train, self.y_test
        )
        
        self.results['NeuralNetwork'] = nn_result
        print("✅ Neural network training complete")
    
    def visualize_results(self):
        """Visualize results"""
        print("\n📘 Results Visualization")
        
        # Model performance comparison
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
        
        print("✅ Visualization complete")
        return top_positive, top_negative
    
    def analyze_results(self, top_positive_words, top_negative_words):
        """Analyze and print results"""
        print("\n📘 Results Analysis")
        
        print_model_summary(self.results)
        print_nlp_insights()
        print_top_features(top_positive_words, top_negative_words)
        print_best_practices()
        
        print("✅ Results analysis complete")
    
    def setup_predictor(self):
        """Setup predictor"""
        self.predictor = ModelPredictor(
            self.traditional_models,
            self.neural_network,
            self.tfidf_extractor,
            self.text_preprocessor
        )
    
    def demo_predictions(self):
        """Demo predictions"""
        print("\n📘 Sample Predictions Demo")
        
        sample_texts = [
            "This product is absolutely fantastic!",
            "Terrible quality, very disappointed.",
            "Good but could be better.",
            "Amazing experience, highly recommend!"
        ]
        
        for text in sample_texts:
            result = self.predictor.predict_sentiment(text)
            print(f"Text: '{result['text']}'")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Model predictions: {result['predictions']}")
            print("-" * 40)
        
        print("✅ Prediction demo complete")
    
    def run(self):
        """Run complete pipeline"""
        print("🚀 English Sentiment Analysis Application Started")
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
            
            print("\n🎉 English Sentiment Analysis Application Complete!")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Error occurred: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function"""
    app = EnglishSentimentAnalysisApp()
    app.run()


if __name__ == "__main__":
    main()