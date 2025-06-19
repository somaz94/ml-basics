import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 필요한 NLTK 데이터 다운로드
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# 합성 감정 분석 데이터셋 생성
print("\n합성 감정 분석 데이터셋 생성 중...")
np.random.seed(42)

# 긍정 및 부정 리뷰 샘플
positive_reviews = [
    "이 제품은 정말 놀라워요! 너무 좋아요.",
    "훌륭한 품질과 좋은 서비스입니다.",
    "멋진 경험, 강력히 추천합니다!",
    "환상적인 제품, 기대를 뛰어넘었어요.",
    "가격 대비 훌륭한 가치, 매우 만족합니다.",
    "뛰어난 성능과 신뢰성입니다.",
    "최고 품질, 최고의 구매였어요!",
    "믿을 수 없는 기능들, 정말 사랑해요.",
    "제 필요에 완벽하고, 훌륭한 선택입니다.",
    "훌륭한 디자인과 기능성입니다."
]

negative_reviews = [
    "끔찍한 제품, 돈 낭비입니다.",
    "끔찍한 품질, 매우 실망했습니다.",
    "끔찍한 경험, 추천하지 않습니다.",
    "나쁜 서비스와 안 좋은 제품입니다.",
    "최악의 구매, 이것은 피하세요.",
    "실망스러운 품질, 가치가 없어요.",
    "나쁜 디자인과 부족한 기능성입니다.",
    "끔찍한 고객 서비스 경험입니다.",
    "쓸모없는 제품, 구매를 후회합니다.",
    "부족한 성능과 신뢰성 문제가 있습니다."
]

# 변형을 추가하여 더 많은 샘플 생성
def generate_reviews(base_reviews, sentiment, count=500):
    reviews = []
    for _ in range(count):
        base = np.random.choice(base_reviews)
        # 변형 추가
        variations = [
            "정말 " + base,
            base + " 확실히!",
            "제 생각에는 " + base,
            base + " 제 의견으로는.",
            "절대적으로 " + base,
            base + " 의심의 여지가 없어요.",
            "분명히 " + base,
            base + " 의문의 여지 없이.",
            "명백히 " + base,
            base + " 확실해요."
        ]
        reviews.append(np.random.choice(variations))
    return reviews

# 데이터셋 생성
positive_samples = generate_reviews(positive_reviews, 1, 500)
negative_samples = generate_reviews(negative_reviews, 0, 500)

# 데이터 결합
texts = positive_samples + negative_samples
labels = [1] * 500 + [0] * 500

# DataFrame 생성
df = pd.DataFrame({
    'text': texts,
    'sentiment': labels
})

print(f"데이터셋 형태: {df.shape}")
print(f"감정 분포: {df['sentiment'].value_counts()}")

# -----------------------
# 텍스트 전처리
# -----------------------
print("\n📘 텍스트 전처리")

def preprocess_text(text):
    # 소문자 변환 (한글은 대소문자 구분이 없으므로 생략)
    # 특수문자와 숫자 제거 (한글, 영문, 공백만 유지)
    text = re.sub(r'[^가-힣a-zA-Z\s]', '', text)
    # 추가 공백 제거
    text = ' '.join(text.split())
    return text

# 전처리 적용
df['processed_text'] = df['text'].apply(preprocess_text)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], df['sentiment'], 
    test_size=0.3, random_state=42, stratify=df['sentiment']
)

print(f"훈련 세트: {len(X_train)}, 테스트 세트: {len(X_test)}")

# -----------------------
# TF-IDF 벡터화
# -----------------------
print("\n📘 TF-IDF 벡터화")

# TF-IDF 벡터라이저 생성
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    stop_words=None,  # 한글 불용어는 별도로 설정 필요
    min_df=2,
    max_df=0.95
)

# 훈련 데이터에 맞추고 변환
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"TF-IDF 특성: {X_train_tfidf.shape[1]}")

# -----------------------
# 전통적 머신러닝 모델
# -----------------------
print("\n📘 전통적 머신러닝 모델")

models = {
    'NaiveBayes': MultinomialNB(),
    'SVM': SVC(kernel='linear', random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\n{name} 학습 중...")
    
    # 모델 학습
    model.fit(X_train_tfidf, y_train)
    
    # 예측
    y_pred = model.predict(X_test_tfidf)
    
    # 평가 지표
    acc = accuracy_score(y_test, y_pred)
    
    results[name] = {
        'accuracy': acc,
        'predictions': y_pred,
        'model': model
    }
    
    print(f"{name} 정확도: {acc:.3f}")

# -----------------------
# 딥러닝: 신경망
# -----------------------
print("\n📘 딥러닝 - 감정 분석을 위한 신경망")

# PyTorch를 위해 희소 행렬을 밀집 행렬로 변환
X_train_dense = X_train_tfidf.toarray()
X_test_dense = X_test_tfidf.toarray()

# 실제 특성 수 가져오기
input_size = X_train_dense.shape[1]
print(f"신경망 입력 크기: {input_size}")

# PyTorch 텐서로 변환
X_train_tensor = torch.FloatTensor(X_train_dense)
y_train_tensor = torch.LongTensor(y_train.values)
X_test_tensor = torch.FloatTensor(X_test_dense)
y_test_tensor = torch.LongTensor(y_test.values)

# 신경망 구축
class SentimentNN(nn.Module):
    def __init__(self, input_size):
        super(SentimentNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)  # 2개 클래스: 부정/긍정
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 실제 입력 크기로 모델 초기화
nn_model = SentimentNN(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

# 신경망 학습
print("신경망 학습 중...")
nn_model.train()
for epoch in range(50):
    optimizer.zero_grad()
    outputs = nn_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'에포크 [{epoch+1}/50], 손실: {loss.item():.4f}')

# 신경망 평가
nn_model.eval()
with torch.no_grad():
    outputs = nn_model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    nn_accuracy = accuracy_score(y_test, predicted.numpy())
    
    results['NeuralNetwork'] = {
        'accuracy': nn_accuracy,
        'predictions': predicted.numpy()
    }
    
    print(f"신경망 정확도: {nn_accuracy:.3f}")

# -----------------------
# 시각화: 결과 분석
# -----------------------
print("\n📘 시각화 - 결과 분석")

# 1. 모델 비교
plt.figure(figsize=(15, 10))

# 정확도 비교
plt.subplot(2, 3, 1)
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
colors = ['skyblue', 'lightcoral', 'lightgreen']

plt.bar(model_names, accuracies, color=colors)
plt.ylabel('정확도')
plt.title('모델 정확도 비교')
plt.ylim(0, 1)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center')

# 2. 혼동행렬
for i, (name, result) in enumerate(results.items()):
    plt.subplot(2, 3, i + 2)
    cm = confusion_matrix(y_test, result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} 혼동행렬')
    plt.xlabel('예측값')
    plt.ylabel('실제값')

plt.tight_layout()
plt.show()

# 3. 긍정 및 부정 감정에 대한 워드클라우드
plt.figure(figsize=(12, 5))

# 긍정 감정 워드클라우드
plt.subplot(1, 2, 1)
positive_text = ' '.join(df[df['sentiment'] == 1]['processed_text'])
wordcloud_positive = WordCloud(
    font_path='/System/Library/Fonts/AppleSDGothicNeo.ttc',  # 한글 폰트 경로 추가
    width=400, height=300, 
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
wordcloud_negative = WordCloud(
    font_path='/System/Library/Fonts/AppleSDGothicNeo.ttc',  # 한글 폰트 경로 추가
    width=400, height=300, 
    background_color='white',
    colormap='Reds',
    max_words=100
).generate(negative_text)

plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.title('부정 감정 워드클라우드')

plt.tight_layout()
plt.show()

# 4. 특성 중요도 (TF-IDF 점수)
plt.figure(figsize=(12, 6))

# 특성 이름 가져오기
feature_names = tfidf_vectorizer.get_feature_names_out()

# 각 클래스에 대한 평균 TF-IDF 점수 계산
positive_indices = df[df['sentiment'] == 1].index
negative_indices = df[df['sentiment'] == 0].index

# 모든 데이터에 대한 TF-IDF 행렬 가져오기
all_tfidf = tfidf_vectorizer.transform(df['processed_text'])

# 평균 점수 계산
positive_scores = all_tfidf[positive_indices].mean(axis=0).A1
negative_scores = all_tfidf[negative_indices].mean(axis=0).A1

# 각 클래스의 상위 특성 가져오기
top_positive_features = np.argsort(positive_scores)[-10:]
top_negative_features = np.argsort(negative_scores)[-10:]

# 상위 긍정 특성 플롯
plt.subplot(1, 2, 1)
top_positive_words = [feature_names[i] for i in top_positive_features]
top_positive_values = [positive_scores[i] for i in top_positive_features]

plt.barh(range(len(top_positive_words)), top_positive_values, color='green', alpha=0.7)
plt.yticks(range(len(top_positive_words)), top_positive_words)
plt.xlabel('평균 TF-IDF 점수')
plt.title('상위 긍정 감정 단어')
plt.gca().invert_yaxis()

# 상위 부정 특성 플롯
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

# -----------------------
# 모델 비교 및 NLP 통찰
# -----------------------
print("\n📘 모델 비교 및 NLP 통찰")
print("=" * 50)
print("모델 성능 요약:")
print("-" * 30)
for name, result in results.items():
    print(f"{name:15} | 정확도: {result['accuracy']:.3f}")
print("=" * 50)

# NLP 해석
print("\n📘 NLP 해석:")
print("- TF-IDF: 특성 추출을 위한 용어 빈도-역문서 빈도")
print("- Naive Bayes: 베이즈 정리에 기반한 확률적 분류기")
print("- SVM: 텍스트 분류를 위한 서포트 벡터 머신")
print("- 신경망: 감정 분석을 위한 딥러닝 접근법")

# 특성 통찰
print(f"\n📘 주요 특성 (상위 5개 긍정 단어):")
for i, word in enumerate(top_positive_words[-5:], 1):
    print(f"{i}. {word}")

print(f"\n📘 주요 특성 (상위 5개 부정 단어):")
for i, word in enumerate(top_negative_words[-5:], 1):
    print(f"{i}. {word}")

print("\n📘 NLP 모범 사례:")
print("- 텍스트 전처리: 소문자 변환, 특수문자 제거")
print("- TF-IDF 벡터화: 단어 중요도 포착")
print("- 불용어 제거: 의미 있는 단어에 집중")
print("- N-gram 특성: 단어 조합 포착")
print("- 교차 검증: 모델 일반화 보장")

# 샘플 예측
print("\n📘 샘플 예측:")
sample_texts = [
    "이 제품은 정말 환상적입니다!",
    "끔찍한 품질, 매우 실망했습니다.",
    "좋지만 더 나을 수 있어요.",
    "놀라운 경험, 강력히 추천합니다!"
]

for text in sample_texts:
    processed = preprocess_text(text)
    tfidf_features = tfidf_vectorizer.transform([processed])
    
    # 모든 모델에서 예측 가져오기
    predictions = {}
    for name, result in results.items():
        if name == 'NeuralNetwork':
            with torch.no_grad():
                # 밀집 배열로 변환하고 올바른 형태 보장
                nn_features = tfidf_features.toarray()
                nn_output = nn_model(torch.FloatTensor(nn_features))
                _, pred = torch.max(nn_output, 1)
                predictions[name] = pred.item()
        else:
            predictions[name] = result['model'].predict(tfidf_features)[0]
    
    sentiment = "긍정" if predictions['NaiveBayes'] == 1 else "부정"
    print(f"텍스트: '{text}'")
    print(f"감정: {sentiment}")
    print(f"모델 예측: {predictions}")
    print("-" * 40) 