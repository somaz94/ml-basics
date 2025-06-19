# 실습9: 감정 분석

<br/>

## 개요 (Overview)
- 텍스트 분류 실습을 위한 합성 감정 분석 데이터셋입니다.
- TF-IDF, Naive Bayes, SVM, 신경망 등 다양한 NLP 기법을 사용하여 텍스트 감정을 긍정 또는 부정으로 분류합니다.

<br/>

## 데이터셋 소개
- **합성 감정 데이터셋**: 1,000개 리뷰 (500개 긍정, 500개 부정)
- 특성: 감정 라벨이 있는 텍스트 리뷰
- 타겟: 이진 분류 (0: 부정, 1: 긍정)
- 텍스트 전처리: 소문자 변환, 특수문자 제거, 공백 정규화

<br/>

## 환경 준비 및 실행법
```bash
# 1. 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 2. 패키지 설치
pip3 install -r requirements.txt

# 3. 실습 코드 실행
python3 sentiment_analysis_kr.py
```

<br/>

## 주요 실습 내용
- 현실적인 감정 패턴을 가진 합성 데이터셋 생성
- 텍스트 전처리 및 정제
- 특성 추출을 위한 TF-IDF 벡터화
- 전통적 ML 모델: Naive Bayes, SVM
- 딥러닝: PyTorch 신경망
- 시각화: 워드클라우드, 특성 중요도, 모델 비교
- NLP 통찰 및 모범 사례

<br/>

## 결과 해석
- TF-IDF: 문서에서 단어의 중요도 포착
- Naive Bayes: 단어 빈도에 기반한 확률적 접근법
- SVM: 고차원 특성의 선형 분류
- 신경망: 텍스트 분류를 위한 딥러닝 접근법
- 워드클라우드: 감정 단어의 시각적 표현

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