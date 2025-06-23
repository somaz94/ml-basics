# 감정 분석 프로젝트 (영어 & 한국어)

이 프로젝트는 머신러닝과 딥러닝 기법을 사용하여 영어와 한국어 텍스트에 대한 감정 분석을 구현합니다. 코드베이스는 완전히 모듈화되어 있으며 별도의 구현으로 두 언어를 모두 지원합니다.

<br/>

## 프로젝트 구조

```
sentiment_analysis/
├── main.py                    # 영어 버전 메인 애플리케이션
├── main_kr.py                 # 한국어 버전 메인 애플리케이션
├── requirements.txt           # 종속성
├── README.md                 # 영어 문서
├── README-KR.md              # 한국어 문서
├── sentiment_analysis.py     # 원본 영어 스크립트 (레거시)
├── sentiment_analysis_kr.py  # 원본 한국어 스크립트 (레거시)
├── src/                      # 영어 버전 모듈화된 코드
│   ├── config.py             # 설정
│   ├── data/                 # 데이터 처리 모듈
│   │   ├── __init__.py
│   │   └── generator.py      # 데이터 생성 및 전처리
│   ├── models/               # ML/DL 모델
│   │   ├── __init__.py
│   │   ├── traditional.py    # 전통적 ML 모델
│   │   └── neural_network.py # 신경망 모델
│   ├── visualization/        # 시각화 모듈
│   │   ├── __init__.py
│   │   └── plots.py          # 플롯 함수
│   └── utils/                # 유틸리티 모듈
│       ├── __init__.py
│       └── helpers.py        # 헬퍼 함수
└── src_kr/                   # 한국어 버전 모듈화된 코드
    ├── config.py             # 한국어 설정
    ├── data/                 # 한국어 데이터 처리
    ├── models/               # 한국어 ML/DL 모델
    ├── visualization/        # 한국어 시각화
    └── utils/                # 한국어 유틸리티
```

<br/>

## 기능

### 1. 이중 언어 지원
- **영어 버전**: 영어 텍스트에 대한 완전한 감정 분석 파이프라인
- **한국어 버전**: 적절한 폰트 지원을 통한 한국어 텍스트 전용 구현

### 2. 데이터 생성 및 전처리
- 합성 감정 데이터셋 생성
- 언어별 텍스트 전처리
- 훈련/테스트 데이터 분할

### 3. 특성 추출
- TF-IDF 벡터화
- N-gram 특성 추출
- 특성 중요도 분석

### 4. 모델 훈련
- **전통적 ML 모델**:
  - Naive Bayes
  - Support Vector Machine (SVM)
- **딥러닝 모델**:
  - 다층 신경망 (PyTorch)

### 5. 시각화
- 모델 성능 비교
- 혼동 행렬
- 워드클라우드 (긍정/부정)
- TF-IDF 특성 중요도
- 데이터 분포 분석

### 6. 예측 시스템
- 단일 텍스트 감정 예측
- 배치 예측ㅋㅋㅋㅋㅋㅋ
- 다수결 투표 시스템

<br/>

## 설치 및 사용법

### 1. 가상환경 생성 및 활성화
```bash
# Python 가상환경 생성
python3 -m venv venv

# 가상환경 활성화 (macOS/Linux)
source venv/bin/activate

# 가상환경 활성화 (Windows)
# venv\Scripts\activate
```

### 2. 종속성 설치
```bash
pip install -r requirements.txt
```

### 3. 애플리케이션 실행

#### 영어 버전
```bash
python main.py
```

#### 한국어 버전
```bash
python main_kr.py
```

<br/>

## 리팩토링 개선사항

### 원본 코드 문제점
- 단일 파일에 모든 코드 집중 (각각 1000줄 이상)
- 함수/클래스 구조 없음
- 하드코딩된 설정 값
- 재사용성과 확장성 부족
- 테스트와 유지보수 어려움

### 리팩토링 후
- **모듈화**: 기능별로 모듈로 분리
- **객체지향 설계**: 클래스 기반 아키텍처
- **설정 관리**: 중앙화된 설정 파일
- **재사용성**: 각 모듈을 독립적으로 사용 가능
- **확장성**: 새로운 모델이나 기능 추가 용이
- **가독성**: 명확하고 이해하기 쉬운 코드 구조
- **유지보수성**: 각 모듈을 독립적으로 수정 가능

<br/>

## 모듈 설명

### 영어 버전 (src/)

#### `src/config.py`
모든 프로젝트 설정을 관리하는 설정 관리자

#### `src/data/generator.py`
- `EnglishSentimentDataGenerator`: 합성 영어 감정 데이터
- `TextPreprocessor`: 영어 텍스트 전처리
- `create_train_test_split`: 데이터 분할

#### `src/models/traditional.py`
- `TfidfFeatureExtractor`: TF-IDF 특성 추출
- `TraditionalModels`: Naive Bayes, SVM 모델 관리

#### `src/models/neural_network.py`
- `SentimentNN`: PyTorch 신경망 모델
- `NeuralNetworkTrainer`: 신경망 훈련

#### `src/visualization/plots.py`
- `SentimentVisualizer`: 모든 시각화 함수

#### `src/utils/helpers.py`
- `ModelPredictor`: 통합 예측 시스템
- 다양한 헬퍼 함수

### 한국어 버전 (src_kr/)
영어 버전과 동일한 구조이지만 한국어 언어 처리에 최적화되어 있습니다.

<br/>

## 사용 예제

### 개별 모듈 사용
```python
# 영어 버전
from src.data import EnglishSentimentDataGenerator, TextPreprocessor
from src.models import TfidfFeatureExtractor, TraditionalModels

# 한국어 버전
from src_kr.data import KoreanSentimentDataGenerator, TextPreprocessor
from src_kr.models import TfidfFeatureExtractor, TraditionalModels

# 데이터 생성
generator = EnglishSentimentDataGenerator()  # 또는 KoreanSentimentDataGenerator()
df = generator.generate_dataset()

# 텍스트 전처리
preprocessor = TextPreprocessor()
df = preprocessor.process_dataframe(df)

# 특성 추출
extractor = TfidfFeatureExtractor()
features = extractor.fit_transform(df['processed_text'])

# 모델 훈련
models = TraditionalModels()
results = models.train_and_evaluate(X_train, X_test, y_train, y_test)
```

### 완전한 파이프라인
```python
# 영어 버전
from main import EnglishSentimentAnalysisApp
app = EnglishSentimentAnalysisApp()
app.run()

# 한국어 버전
from main_kr import KoreanSentimentAnalysisApp
app = KoreanSentimentAnalysisApp()
app.run()
```

<br/>

## 확장 가능성

이 리팩토링된 구조는 쉬운 확장을 가능하게 합니다:

1. **새로운 모델**: `src/models/` 또는 `src_kr/models/`에 새로운 모델 모듈 추가
2. **다른 언어**: 새로운 언어별 디렉토리 생성
3. **새로운 시각화**: 시각화 모듈에 새로운 플롯 함수 추가
4. **API 서버**: FastAPI 또는 Flask로 REST API 생성
5. **데이터베이스 통합**: 실제 데이터 소스에 연결
6. **모델 지속성**: 모델 저장/로딩 기능 추가
7. **실시간 예측**: 스트리밍 데이터 처리

<br/>

## 성능 벤치마크

두 버전 모두 동일한 알고리즘과 하이퍼파라미터를 사용하여 일관된 성능을 보장합니다:

- Naive Bayes: ~95% 정확도
- SVM: ~96% 정확도  
- 신경망: ~97% 정확도

<br/>

## 주요 학습 포인트
- **텍스트 전처리**: NLP 작업에 필수적인 단계
- **TF-IDF 벡터화**: 텍스트를 수치적 특성으로 변환
- **NLP 모델**: 텍스트 분류의 다양한 접근법 이해
- **특성 중요도**: 감정 분석을 위한 주요 단어 식별

<br/>

## 기술적 특징
- **TF-IDF**: 용어 빈도-역문서 빈도
- **N-gram**: 단어 조합 (단일어와 이중어)
- **불용어**: 감정을 전달하지 않는 일반적인 단어 제거
- **워드클라우드**: 단어 빈도의 시각적 표현

<br/>

## NLP 맥락
- **긍정 (1)**: 긍정적 감정 - 행복, 만족, 좋음
- **부정 (0)**: 부정적 감정 - 불행, 실망, 나쁨
- **텍스트 전처리**: 일관된 특성 추출을 위해 필수
- **특성 공학**: 텍스트를 기계가 읽을 수 있는 형식으로 변환

<br/>

## 마무리 (Conclusion)
- 현실적인 텍스트 분류 데이터셋으로 실습
- 전통적 ML과 딥러닝 접근법 비교
- NLP 전처리 및 특성 추출 이해
- 텍스트 분석을 위한 시각화 기법 학습

<br/>

## 참고 (Reference)
- [TF-IDF Vectorization](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [NLTK Documentation](https://www.nltk.org/)
- [WordCloud Library](https://amueller.github.io/word_cloud/)
- [Text Classification Best Practices](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) 