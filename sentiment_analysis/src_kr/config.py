"""
프로젝트 설정 파일 (한국어 버전)
"""

# 데이터 설정
DATA_CONFIG = {
    'test_size': 0.3,
    'random_state': 42,
    'sample_size': 500
}

# TF-IDF 설정
TFIDF_CONFIG = {
    'max_features': 1000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.95,
    'stop_words': None  # 한글 불용어는 별도 설정 필요
}

# 신경망 설정
NN_CONFIG = {
    'hidden_sizes': [256, 128, 64],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'epochs': 50,
    'num_classes': 2
}

# 시각화 설정
VIZ_CONFIG = {
    'font_family': 'AppleGothic',
    'font_path': '/System/Library/Fonts/AppleSDGothicNeo.ttc',
    'figure_size': (15, 10),
    'wordcloud_size': (400, 300)
}

# 모델 설정
MODEL_CONFIG = {
    'svm_kernel': 'linear',
    'random_state': 42
}
