# Practice 9: Sentiment Analysis

<br/>

## Overview
- Synthetic sentiment analysis dataset for text classification practice.
- Use various NLP techniques including TF-IDF, Naive Bayes, SVM, and Neural Network to classify text sentiment as positive or negative.

<br/>

## Dataset Introduction
- **Synthetic Sentiment Dataset**: 1,000 reviews (500 positive, 500 negative)
- Features: Text reviews with sentiment labels
- Target: Binary classification (0: Negative, 1: Positive)
- Text preprocessing: Lowercase, special character removal, whitespace normalization

<br/>

## Environment Setup and Execution
```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install packages
pip3 install -r requirements.txt

# 3. Run practice code
python3 sentiment_analysis.py
```

<br/>

## Main Practice Contents
- Synthetic dataset generation with realistic sentiment patterns
- Text preprocessing and cleaning
- TF-IDF vectorization for feature extraction
- Traditional ML models: Naive Bayes, SVM
- Deep Learning: Neural Network with PyTorch
- Visualization: Word clouds, feature importance, model comparison
- NLP insights and best practices

<br/>

## Result Interpretation
- TF-IDF: Captures word importance in documents
- Naive Bayes: Probabilistic approach based on word frequencies
- SVM: Linear classification with high-dimensional features
- Neural Network: Deep learning approach for text classification
- Word clouds: Visual representation of sentiment words

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