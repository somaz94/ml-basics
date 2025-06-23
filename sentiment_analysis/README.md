# Sentiment Analysis Project (English & Korean)

This project implements sentiment analysis for both English and Korean text using machine learning and deep learning techniques. The codebase is completely modularized and supports both languages with separate implementations.

## Project Structure

```
sentiment_analysis/
├── main.py                    # English version main application
├── main_kr.py                 # Korean version main application
├── requirements.txt           # Dependencies
├── README.md                 # English documentation
├── README-KR.md              # Korean documentation
├── sentiment_analysis.py     # Original English script (legacy)
├── sentiment_analysis_kr.py  # Original Korean script (legacy)
├── src/                      # English version modularized code
│   ├── config.py             # Configuration
│   ├── data/                 # Data processing modules
│   │   ├── __init__.py
│   │   └── generator.py      # Data generation and preprocessing
│   ├── models/               # ML/DL models
│   │   ├── __init__.py
│   │   ├── traditional.py    # Traditional ML models
│   │   └── neural_network.py # Neural network model
│   ├── visualization/        # Visualization modules
│   │   ├── __init__.py
│   │   └── plots.py          # Plotting functions
│   └── utils/                # Utility modules
│       ├── __init__.py
│       └── helpers.py        # Helper functions
└── src_kr/                   # Korean version modularized code
    ├── config.py             # Korean configuration
    ├── data/                 # Korean data processing
    ├── models/               # Korean ML/DL models
    ├── visualization/        # Korean visualization
    └── utils/                # Korean utilities
```

## Features

### 1. Dual Language Support
- **English Version**: Complete sentiment analysis pipeline for English text
- **Korean Version**: Specialized implementation for Korean text with proper font support

### 2. Data Generation & Preprocessing
- Synthetic sentiment dataset generation
- Language-specific text preprocessing
- Train/test data splitting

### 3. Feature Extraction
- TF-IDF vectorization
- N-gram feature extraction
- Feature importance analysis

### 4. Model Training
- **Traditional ML Models**:
  - Naive Bayes
  - Support Vector Machine (SVM)
- **Deep Learning Model**:
  - Multi-layer Neural Network (PyTorch)

### 5. Visualization
- Model performance comparison
- Confusion matrices
- Word clouds (positive/negative)
- TF-IDF feature importance
- Data distribution analysis

### 6. Prediction System
- Single text sentiment prediction
- Batch prediction
- Majority voting system

## Installation and Usage

### 1. Create and Activate Virtual Environment
```bash
# Create Python virtual environment
python3 -m venv venv

# Activate virtual environment (macOS/Linux)
source venv/bin/activate

# Activate virtual environment (Windows)
# venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Applications

#### English Version
```bash
python main.py
```

#### Korean Version
```bash
python main_kr.py
```

## Refactoring Improvements

### Original Code Issues
- All code concentrated in single files (1000+ lines each)
- No function/class structure
- Hard-coded configuration values
- Poor reusability and extensibility
- Difficult to test and maintain

### After Refactoring
- **Modularization**: Separated by functionality into modules
- **Object-Oriented Design**: Class-based architecture
- **Configuration Management**: Centralized config files
- **Reusability**: Each module can be used independently
- **Extensibility**: Easy to add new models or features
- **Readability**: Clear and understandable code structure
- **Maintainability**: Each module can be modified independently

## Module Description

### English Version (src/)

#### `src/config.py`
Configuration management for all project settings.

#### `src/data/generator.py`
- `EnglishSentimentDataGenerator`: Synthetic English sentiment data
- `TextPreprocessor`: English text preprocessing
- `create_train_test_split`: Data splitting

#### `src/models/traditional.py`
- `TfidfFeatureExtractor`: TF-IDF feature extraction
- `TraditionalModels`: Naive Bayes, SVM model management

#### `src/models/neural_network.py`
- `SentimentNN`: PyTorch neural network model
- `NeuralNetworkTrainer`: Neural network training

#### `src/visualization/plots.py`
- `SentimentVisualizer`: All visualization functions

#### `src/utils/helpers.py`
- `ModelPredictor`: Unified prediction system
- Various helper functions

### Korean Version (src_kr/)
Same structure as English version but optimized for Korean language processing.

## Usage Examples

### Individual Module Usage
```python
# English version
from src.data import EnglishSentimentDataGenerator, TextPreprocessor
from src.models import TfidfFeatureExtractor, TraditionalModels

# Korean version
from src_kr.data import KoreanSentimentDataGenerator, TextPreprocessor
from src_kr.models import TfidfFeatureExtractor, TraditionalModels

# Generate data
generator = EnglishSentimentDataGenerator()  # or KoreanSentimentDataGenerator()
df = generator.generate_dataset()

# Text preprocessing
preprocessor = TextPreprocessor()
df = preprocessor.process_dataframe(df)

# Feature extraction
extractor = TfidfFeatureExtractor()
features = extractor.fit_transform(df['processed_text'])

# Model training
models = TraditionalModels()
results = models.train_and_evaluate(X_train, X_test, y_train, y_test)
```

### Complete Pipeline
```python
# English version
from main import EnglishSentimentAnalysisApp
app = EnglishSentimentAnalysisApp()
app.run()

# Korean version
from main_kr import KoreanSentimentAnalysisApp
app = KoreanSentimentAnalysisApp()
app.run()
```

## Extension Possibilities

This refactored structure facilitates easy extensions:

1. **New Models**: Add new model modules to `src/models/` or `src_kr/models/`
2. **Other Languages**: Create new language-specific directories
3. **New Visualizations**: Add new plot functions to visualization modules
4. **API Server**: Create REST API with FastAPI or Flask
5. **Database Integration**: Connect to real data sources
6. **Model Persistence**: Add model saving/loading functionality
7. **Real-time Prediction**: Process streaming data

## Performance Benchmarks

Both versions use identical algorithms and hyperparameters, ensuring consistent performance:

- Naive Bayes: ~95% accuracy
- SVM: ~96% accuracy  
- Neural Network: ~97% accuracy

<br/>

## Key Learning Points
- **Text Preprocessing**: Essential steps for NLP tasks
- **TF-IDF Vectorization**: Converting text to numerical features
- **NLP Models**: Understanding different approaches to text classification
- **Feature Importance**: Identifying key words for sentiment analysis

<br/>

## Technical Features
- **TF-IDF**: Term frequency-inverse document frequency
- **N-grams**: Word combinations (unigrams and bigrams)
- **Stop Words**: Removing common words that don't carry sentiment
- **Word Clouds**: Visual representation of word frequencies

<br/>

## NLP Context
- **Positive (1)**: Positive sentiment - happy, satisfied, good
- **Negative (0)**: Negative sentiment - unhappy, disappointed, bad
- **Text Preprocessing**: Essential for consistent feature extraction
- **Feature Engineering**: Converting text to machine-readable format

<br/>

## Conclusion
- Practice with realistic text classification dataset
- Compare traditional ML and deep learning approaches
- Understand NLP preprocessing and feature extraction
- Learn visualization techniques for text analysis

<br/>

## Reference
- [TF-IDF Vectorization](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [NLTK Documentation](https://www.nltk.org/)
- [WordCloud Library](https://amueller.github.io/word_cloud/)
- [Text Classification Best Practices](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) 