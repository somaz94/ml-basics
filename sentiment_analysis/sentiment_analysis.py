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

# Download required NLTK data
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

# Generate synthetic sentiment dataset
print("\nGenerating synthetic sentiment analysis dataset...")
np.random.seed(42)

# Sample positive and negative reviews
positive_reviews = [
    "This product is amazing! I love it so much.",
    "Excellent quality and great service.",
    "Wonderful experience, highly recommended!",
    "Fantastic product, exceeded my expectations.",
    "Great value for money, very satisfied.",
    "Outstanding performance and reliability.",
    "Superb quality, best purchase ever!",
    "Incredible features, absolutely love it.",
    "Perfect for my needs, excellent choice.",
    "Brilliant design and functionality."
]

negative_reviews = [
    "Terrible product, complete waste of money.",
    "Awful quality, very disappointed.",
    "Horrible experience, would not recommend.",
    "Poor service and bad product.",
    "Worst purchase ever, avoid this.",
    "Disappointing quality, not worth it.",
    "Bad design and poor functionality.",
    "Terrible customer service experience.",
    "Useless product, regret buying it.",
    "Poor performance and reliability issues."
]

# Generate more samples with variations
def generate_reviews(base_reviews, sentiment, count=500):
    reviews = []
    for _ in range(count):
        base = np.random.choice(base_reviews)
        # Add some variation
        variations = [
            "Really " + base,
            base + " Definitely!",
            "I think " + base,
            base + " In my opinion.",
            "Absolutely " + base,
            base + " No doubt about it.",
            "Clearly " + base,
            base + " Without question.",
            "Obviously " + base,
            base + " For sure."
        ]
        reviews.append(np.random.choice(variations))
    return reviews

# Generate dataset
positive_samples = generate_reviews(positive_reviews, 1, 500)
negative_samples = generate_reviews(negative_reviews, 0, 500)

# Combine data
texts = positive_samples + negative_samples
labels = [1] * 500 + [0] * 500

# Create DataFrame
df = pd.DataFrame({
    'text': texts,
    'sentiment': labels
})

print(f"Dataset shape: {df.shape}")
print(f"Sentiment distribution: {df['sentiment'].value_counts()}")

# -----------------------
# Text Preprocessing
# -----------------------
print("\n📘 Text Preprocessing")

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Apply preprocessing
df['processed_text'] = df['text'].apply(preprocess_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], df['sentiment'], 
    test_size=0.3, random_state=42, stratify=df['sentiment']
)

print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

# -----------------------
# TF-IDF Vectorization
# -----------------------
print("\n📘 TF-IDF Vectorization")

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=2,
    max_df=0.95
)

# Fit and transform training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"TF-IDF features: {X_train_tfidf.shape[1]}")

# -----------------------
# Traditional ML Models
# -----------------------
print("\n📘 Traditional Machine Learning Models")

models = {
    'NaiveBayes': MultinomialNB(),
    'SVM': SVC(kernel='linear', random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train_tfidf, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    
    results[name] = {
        'accuracy': acc,
        'predictions': y_pred,
        'model': model
    }
    
    print(f"{name} Accuracy: {acc:.3f}")

# -----------------------
# Deep Learning: Neural Network
# -----------------------
print("\n📘 Deep Learning - Neural Network for Sentiment Analysis")

# Convert sparse matrix to dense for PyTorch
X_train_dense = X_train_tfidf.toarray()
X_test_dense = X_test_tfidf.toarray()

# Get actual feature count
input_size = X_train_dense.shape[1]
print(f"Neural Network input size: {input_size}")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_dense)
y_train_tensor = torch.LongTensor(y_train.values)
X_test_tensor = torch.FloatTensor(X_test_dense)
y_test_tensor = torch.LongTensor(y_test.values)

# Build neural network
class SentimentNN(nn.Module):
    def __init__(self, input_size):
        super(SentimentNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)  # 2 classes: negative/positive
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize model with actual input size
nn_model = SentimentNN(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

# Train neural network
print("Training Neural Network...")
nn_model.train()
for epoch in range(50):
    optimizer.zero_grad()
    outputs = nn_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/50], Loss: {loss.item():.4f}')

# Evaluate neural network
nn_model.eval()
with torch.no_grad():
    outputs = nn_model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    nn_accuracy = accuracy_score(y_test, predicted.numpy())
    
    results['NeuralNetwork'] = {
        'accuracy': nn_accuracy,
        'predictions': predicted.numpy()
    }
    
    print(f"Neural Network Accuracy: {nn_accuracy:.3f}")

# -----------------------
# Visualization: Results Analysis
# -----------------------
print("\n📘 Visualization - Results Analysis")

# 1. Model comparison
plt.figure(figsize=(15, 10))

# Accuracy comparison
plt.subplot(2, 3, 1)
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
colors = ['skyblue', 'lightcoral', 'lightgreen']

plt.bar(model_names, accuracies, color=colors)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center')

# 2. Confusion matrices
for i, (name, result) in enumerate(results.items()):
    plt.subplot(2, 3, i + 2)
    cm = confusion_matrix(y_test, result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# 3. Word clouds for positive and negative sentiments
plt.figure(figsize=(12, 5))

# Positive sentiment word cloud
plt.subplot(1, 2, 1)
positive_text = ' '.join(df[df['sentiment'] == 1]['processed_text'])
wordcloud_positive = WordCloud(
    width=400, height=300, 
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
wordcloud_negative = WordCloud(
    width=400, height=300, 
    background_color='white',
    colormap='Reds',
    max_words=100
).generate(negative_text)

plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Sentiment Word Cloud')

plt.tight_layout()
plt.show()

# 4. Feature importance (TF-IDF scores)
plt.figure(figsize=(12, 6))

# Get feature names
feature_names = tfidf_vectorizer.get_feature_names_out()

# Calculate average TF-IDF scores for each class
positive_indices = df[df['sentiment'] == 1].index
negative_indices = df[df['sentiment'] == 0].index

# Get TF-IDF matrix for all data
all_tfidf = tfidf_vectorizer.transform(df['processed_text'])

# Calculate average scores
positive_scores = all_tfidf[positive_indices].mean(axis=0).A1
negative_scores = all_tfidf[negative_indices].mean(axis=0).A1

# Get top features for each class
top_positive_features = np.argsort(positive_scores)[-10:]
top_negative_features = np.argsort(negative_scores)[-10:]

# Plot top positive features
plt.subplot(1, 2, 1)
top_positive_words = [feature_names[i] for i in top_positive_features]
top_positive_values = [positive_scores[i] for i in top_positive_features]

plt.barh(range(len(top_positive_words)), top_positive_values, color='green', alpha=0.7)
plt.yticks(range(len(top_positive_words)), top_positive_words)
plt.xlabel('Average TF-IDF Score')
plt.title('Top Positive Sentiment Words')
plt.gca().invert_yaxis()

# Plot top negative features
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

# -----------------------
# Model Comparison and NLP Insights
# -----------------------
print("\n📘 Model Comparison and NLP Insights")
print("=" * 50)
print("Model Performance Summary:")
print("-" * 30)
for name, result in results.items():
    print(f"{name:15} | Accuracy: {result['accuracy']:.3f}")
print("=" * 50)

# NLP interpretation
print("\n📘 NLP Interpretation:")
print("- TF-IDF: Term frequency-inverse document frequency for feature extraction")
print("- Naive Bayes: Probabilistic classifier based on Bayes theorem")
print("- SVM: Support Vector Machine for text classification")
print("- Neural Network: Deep learning approach for sentiment analysis")

# Feature insights
print(f"\n📘 Key Features (Top 5 Positive Words):")
for i, word in enumerate(top_positive_words[-5:], 1):
    print(f"{i}. {word}")

print(f"\n📘 Key Features (Top 5 Negative Words):")
for i, word in enumerate(top_negative_words[-5:], 1):
    print(f"{i}. {word}")

print("\n📘 NLP Best Practices:")
print("- Text preprocessing: Lowercase, remove special characters")
print("- TF-IDF vectorization: Capture word importance")
print("- Stop words removal: Focus on meaningful words")
print("- N-gram features: Capture word combinations")
print("- Cross-validation: Ensure model generalization")

# Sample predictions
print("\n📘 Sample Predictions:")
sample_texts = [
    "This product is absolutely fantastic!",
    "Terrible quality, very disappointed.",
    "Good but could be better.",
    "Amazing experience, highly recommend!"
]

for text in sample_texts:
    processed = preprocess_text(text)
    tfidf_features = tfidf_vectorizer.transform([processed])
    
    # Get predictions from all models
    predictions = {}
    for name, result in results.items():
        if name == 'NeuralNetwork':
            with torch.no_grad():
                # Convert to dense array and ensure correct shape
                nn_features = tfidf_features.toarray()
                nn_output = nn_model(torch.FloatTensor(nn_features))
                _, pred = torch.max(nn_output, 1)
                predictions[name] = pred.item()
        else:
            predictions[name] = result['model'].predict(tfidf_features)[0]
    
    sentiment = "Positive" if predictions['NaiveBayes'] == 1 else "Negative"
    print(f"Text: '{text}'")
    print(f"Sentiment: {sentiment}")
    print(f"Model predictions: {predictions}")
    print("-" * 40) 