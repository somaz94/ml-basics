"""
Project configuration for English version
"""

# Data configuration
DATA_CONFIG = {
    'test_size': 0.3,
    'random_state': 42,
    'sample_size': 500
}

# TF-IDF configuration
TFIDF_CONFIG = {
    'max_features': 1000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.95,
    'stop_words': 'english'
}

# Neural Network configuration
NN_CONFIG = {
    'hidden_sizes': [256, 128, 64],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'epochs': 50,
    'num_classes': 2
}

# Visualization configuration
VIZ_CONFIG = {
    'font_family': 'DejaVu Sans',
    'font_path': None,
    'figure_size': (15, 10),
    'wordcloud_size': (400, 300)
}

# Model configuration
MODEL_CONFIG = {
    'svm_kernel': 'linear',
    'random_state': 42
}
