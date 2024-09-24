#!/usr/bin/env python
# coding: utf-8

#  

# 주요 어휘 추출 전처리 후 페이지 추천 기능 적용

#   

# 1. 전처리 코드 완성(디버깅 포함, 워드클라우드 생성)

# ※ 라이브러리 및 프레임워크 필요시 설치

# !pip install konlpy

# !pip install JPype1

# !pip install wordcloud matplotlib

# !pip install --upgrade pip

# !pip install --upgrade Pillow

# In[ ]:


import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[ ]:


# 데이터 로드
df = pd.read_csv("data/wanted_crawling_all_data.csv", index_col=None, 
                 parse_dates=['Title', 'URL'], encoding='cp949')


# In[ ]:


# 불용어 리스트를 파일에서 읽어오기
with open('stopwords-ko.txt', 'r', encoding='utf-8') as file:
    stopwords = file.read().splitlines()


# In[ ]:


# 형태소 분석기 초기화
okt = Okt()


# In[ ]:


# 전처리 함수 정의 (명사 추출 및 불용어 제거, 영어 포함)
def preprocess_text(text, okt):
    if isinstance(text, str):
        text = text.lower()  # 소문자로 변환 (영어에 유용)
        text = re.sub(r'\d+', '', text)  # 숫자 제거
        text = re.sub(r'[^\w\s]', '', text)  # 특수 문자 제거
        # 명사 추출
        nouns = okt.nouns(text)
        # 명사와 함께 영어 단어도 추출하기 위해 영어 필터링 추가
        english_words = re.findall(r'\b[a-zA-Z]+\b', text)  # 영어 단어 추출
        # 불용어 제거 및 단어 필터링
        filtered_nouns = [noun for noun in nouns if noun not in stopwords and len(noun) > 1]
        filtered_english = [word for word in english_words if word not in stopwords]
        # 한국어 명사와 영어 단어를 결합하여 반환
        return ' '.join(filtered_nouns + filtered_english)
    return ''


# In[ ]:


# TF-IDF 기반 중요 단어 추출 함수
def get_important_words(column_text, n=50):
    vectorizer = TfidfVectorizer(max_features=n, max_df=0.85, min_df=1) 
    X = vectorizer.fit_transform(column_text)
    if X.shape[0] == 0:  # 문서가 없는 경우
        return []
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = X.toarray().sum(axis=0)
    word_score_pairs = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
    return word_score_pairs


# In[ ]:


# 각 열에 대해 전처리 및 TF-IDF 기반 중요 단어 추출 (URL 칼럼 제외)
important_words_dict = {}


# In[ ]:


for column in df.columns:
    # 해당 칼럼은 처리하지 않음
    if column in ['Title', 'Company', 'Career', 'Deadline', 'Location', 'Duty', 'URL']:
        continue
    if df[column].dtype == 'object':  # 문자열 데이터에 대해서만 처리
        # 전처리 및 명사 추출
        df[column] = df[column].apply(preprocess_text)
        column_text = df[column].dropna().tolist()  # NaN 값 제거 및 리스트로 변환
        
        # 텍스트 샘플 출력 (디버깅용)
        print(f"Column: {column}")
        print("Sample Texts:")
        print(column_text[:5])  # 상위 5개 텍스트 샘플 출력
        
        if len(column_text) > 0:  # 데이터가 있는 경우에만 처리
            # TF-IDF 기반 중요 단어 추출
            important_words = get_important_words(column_text, n=50)
            if important_words:  # 중요 단어가 추출된 경우에만 처리
                important_words_dict[column] = important_words


# In[ ]:


# 각 칼럼별 중요 단어를 CSV 파일로 저장
for column, words in important_words_dict.items():
    df_words = pd.DataFrame(words, columns=['Word', 'Score'])
    df_words.to_csv(f'{column}_important_words.csv', index=False, encoding='utf-8-sig')


# In[ ]:


# 전처리된 텍스트 확인 (디버깅용)
for column in df.columns:
    if column in important_words_dict:
        print(f"Column: {column}")
        print(df[column].head())


# In[ ]:


# 전처리된 데이터프레임 저장
df.to_csv('preprocessed_data_all.csv', index=False, encoding='utf-8-sig')


# In[ ]:


# 워드클라우드 생성 및 저장 함수
def create_wordcloud(words, column_name):
    word_freq = dict(words)  # 중요 단어와 점수를 딕셔너리 형태로 변환
    wordcloud = WordCloud(font_path='C:/Windows/Fonts/malgun.ttf',  # 한글 폰트 경로
                          width=800, height=400, 
                          background_color='white').generate_from_frequencies(word_freq)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {column_name}')
    plt.savefig(f'{column_name}_wordcloud.png', format='png')
    plt.show()

# 각 칼럼별 워드클라우드 생성 및 저장
for column, words in important_words_dict.items():
    create_wordcloud(words, column)


#  

#  

#  

# 2. 데이터 전처리 후 추천기능 적용(통합검색 추천)

#  

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


# In[ ]:


# 데이터 로드
df = pd.read_csv("data/wanted_crawling_all_data.csv", index_col=None, encoding='cp949')


# In[ ]:


# 불용어 리스트를 파일에서 읽어오기
with open('stopwords-ko.txt', 'r', encoding='utf-8') as file:
    stopwords = file.read().splitlines()

# 전처리 함수 정의 (명사 추출 및 불용어 제거, 영어 포함)
def preprocess_text(text, okt):
    if isinstance(text, str):
        text = text.lower()  # 소문자로 변환 (영어에 유용)
        text = re.sub(r'\d+', '', text)  # 숫자 제거
        text = re.sub(r'[^\w\s]', '', text)  # 특수 문자 제거
        # 명사 추출
        nouns = okt.nouns(text)
        # 명사와 함께 영어 단어도 추출하기 위해 영어 필터링 추가
        english_words = re.findall(r'\b[a-zA-Z]+\b', text)  # 영어 단어 추출
        # 불용어 제거 및 단어 필터링
        filtered_nouns = [noun for noun in nouns if noun not in stopwords and len(noun) > 1]
        filtered_english = [word for word in english_words if word not in stopwords]
        # 한국어 명사와 영어 단어를 결합하여 반환
        return ' '.join(filtered_nouns + filtered_english)
    return ''


# In[ ]:


# 형태소 분석기 초기화
from konlpy.tag import Okt
okt = Okt()


# In[ ]:


# 사용자의 조건을 입력받아 유사한 채용 공고 추천하는 함수
def recommend_jobs(user_query, df, okt, top_n=5):
    # NaN 값을 빈 문자열로 대체
    df.fillna('', inplace=True)
    # 여러 칼럼을 전처리 (Title, Company, 주요 업무, 기술 스택 등 필요한 칼럼 추가)
    df['preprocessed_title'] = df['Title']
    df['preprocessed_company'] = df['Company']
    df['preprocessed_career'] = df['Career']
    df['preprocessed_work'] = df['Work'].apply(lambda x: preprocess_text(x, okt))
    df['preprocessed_qualification'] = df['Qualification'].apply(lambda x: preprocess_text(x, okt))
    df['preprocessed_addition'] = df['Addition'].apply(lambda x: preprocess_text(x, okt))
    df['preprocessed_welfare'] = df['Welfare'].apply(lambda x: preprocess_text(x, okt))
    df['preprocessed_skill'] = df['Skill']
    df['preprocessed_tag'] = df['Tag']
    df['preprocessed_deadline'] = df['Deadline']
    df['preprocessed_location'] = df['Location']
    df['preprocessed_duty'] = df['Duty']
    
    # 사용자가 입력한 질의를 전처리
    query_preprocessed = preprocess_text(user_query, okt)

    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer()

    # 여러 칼럼을 하나의 텍스트로 결합 (공백을 추가하여 결합)
    df['combined_text'] = df['preprocessed_title'] + ' ' + df['preprocessed_company'] + ' ' + df['preprocessed_career'] + ' ' + df['preprocessed_work'] + ' ' + df['preprocessed_qualification'] + ' ' + df['preprocessed_addition'] + ' ' + df['preprocessed_welfare'] + ' ' + df['preprocessed_skill'] + ' ' + df['preprocessed_tag'] + ' ' + df['preprocessed_deadline'] + ' ' + df['preprocessed_location'] + ' ' + df['preprocessed_duty']
    combined_matrix = vectorizer.fit_transform(df['combined_text'])
    
    # 사용자의 질의를 벡터화
    query_vector = vectorizer.transform([query_preprocessed])

    # 코사인 유사도 계산
    similarity_scores = cosine_similarity(query_vector, combined_matrix).flatten()

    # 유사도 순으로 상위 N개의 인덱스 추출
    top_indices = similarity_scores.argsort()[-top_n:][::-1]

    # 상위 N개의 회사, 제목, URL을 추천
    recommendations = df[['Title', 'Company', 'URL']].iloc[top_indices]
    return recommendations


# In[ ]:


# 사용자가 선택한 조건
user_query = "데이터 분석"

# 채용 공고 추천
recommended_jobs = recommend_jobs(user_query, df, okt, top_n=5)


# In[ ]:


# 추천 결과 출력
print(recommended_jobs)


#  

#  

#  

# 3. 문답 형식에 따른 채용공고 추천 코드(조건에 따른 검색 추천)

#  

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


# In[ ]:


# 데이터 로드
df = pd.read_csv("data/wanted_crawling_all_data.csv", index_col=None, encoding='cp949')


# In[ ]:


# 불용어 리스트를 파일에서 읽어오기
with open('stopwords-ko.txt', 'r', encoding='utf-8') as file:
    stopwords = file.read().splitlines()

# 전처리 함수 정의 (명사 추출 및 불용어 제거, 영어 포함)
def preprocess_text(text, okt):
    if isinstance(text, str):
        text = text.lower()  # 소문자로 변환
        text = re.sub(r'\d+', '', text)  # 숫자 제거
        text = re.sub(r'[^\w\s]', '', text)  # 특수 문자 제거
        # 명사 추출
        nouns = okt.nouns(text)
        # 영어 단어 추출
        english_words = re.findall(r'\b[a-zA-Z]+\b', text)
        # 불용어 제거 및 단어 필터링
        filtered_nouns = [noun for noun in nouns if noun not in stopwords and len(noun) > 1]
        filtered_english = [word for word in english_words if word not in stopwords]
        # 한국어 명사와 영어 단어를 결합하여 반환
        return ' '.join(filtered_nouns + filtered_english)
    return ''


# In[ ]:


# 형태소 분석기 초기화
from konlpy.tag import Okt
okt = Okt()


# In[ ]:


# 사용자 검색어 입력
def ask_question(question):
    response = input(question + " ")
    return response

# 조건에 따른 필터링
def filter_jobs_by_criteria(df, location=None, duty=None, work=None, qualification=None, addition=None, welfare=None, skill=None, career=None):
    filtered_df = df.copy()
    
    # 조건에 따라 데이터 필터링
    if location and not pd.isna(location):
        filtered_df = filtered_df[filtered_df['Location'].str.strip().str.contains(location.strip(), case=False, na=False)]
    if duty and not pd.isna(duty):
        filtered_df = filtered_df[filtered_df['Duty'].str.strip().str.contains(duty.strip(), case=False, na=False)]
    if work and not pd.isna(work):
        filtered_df = filtered_df[filtered_df['Work'].str.strip().str.contains(work.strip(), case=False, na=False)]
    if qualification and not pd.isna(qualification):
        filtered_df = filtered_df[filtered_df['Qualification'].str.strip().str.contains(qualification.strip(), case=False, na=False)]
    if addition and not pd.isna(addition):
        filtered_df = filtered_df[filtered_df['Addition'].str.strip().str.contains(addition.strip(), case=False, na=False)]
    if welfare and not pd.isna(welfare):
        filtered_df = filtered_df[filtered_df['Welfare'].str.strip().str.contains(welfare.strip(), case=False, na=False)]
    if skill and not pd.isna(skill):
        filtered_df = filtered_df[filtered_df['Skill'].str.strip().str.contains(skill.strip(), case=False, na=False)]
    if career and not pd.isna(career):
        filtered_df = filtered_df[filtered_df['Career'].str.strip().str.contains(career.strip(), case=False, na=False)]
    
    return filtered_df


# 유사 채용 공고 추천 함수 (문답에 따른 범위 좁히기)
def recommend_interactively(df, okt):
    # 사용자의 선택 조건을 순차적으로 입력
    location = ask_question("선호하는 지역을 말씀해주세요:")
    duty = ask_question("선호하는 직무를 말씀해주세요:")
    work = ask_question("주요 업무에서 원하는 조건을 말씀해주세요:")
    qualification = ask_question("자격 사항에서 원하는 조건을 말씀해주세요:")
    addition = ask_question("우대 사항에서 원하는 조건을 말씀해주세요:")
    welfare = ask_question("복지 사항에서 원하는 조건을 말씀해주세요:")
    skill = ask_question("필요한 스킬을 말씀해주세요:")
    career = ask_question("경력에 대한 조건을 말씀해주세요:")

    # 필터링된 데이터프레임
    filtered_df = filter_jobs_by_criteria(df, location, duty, work, qualification, addition, welfare, skill, career)
    
    if filtered_df.empty:
        print("해당 조건에 맞는 채용 공고가 없습니다.")
        return
    
    # 전처리 (필요한 칼럼에 대한 전처리 수행)
    filtered_df['preprocessed_work'] = filtered_df['Work'].apply(lambda x: preprocess_text(x, okt))
    filtered_df['preprocessed_qualification'] = filtered_df['Qualification'].apply(lambda x: preprocess_text(x, okt))
    filtered_df['preprocessed_addition'] = filtered_df['Addition'].apply(lambda x: preprocess_text(x, okt))
    filtered_df['preprocessed_welfare'] = filtered_df['Welfare'].apply(lambda x: preprocess_text(x, okt))

    # 사용자 질의를 입력받고 전처리
    user_query = ask_question("검색하려는 키워드를 입력해주세요:")
    query_preprocessed = preprocess_text(user_query, okt)

    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer()

    # 여러 칼럼을 결합하여 벡터화
    filtered_df['combined_text'] = (
    filtered_df['Title'].fillna('') + ' ' +
    filtered_df['Company'].fillna('') + ' ' +
    filtered_df['preprocessed_work'].fillna('') + ' ' +
    filtered_df['preprocessed_qualification'].fillna('') + ' ' +
    filtered_df['preprocessed_addition'].fillna('') + ' ' +
    filtered_df['preprocessed_welfare'].fillna('') + ' ' +
    filtered_df['Skill'].fillna('') + ' ' +
    filtered_df['Tag'].fillna('') + ' ' +
    filtered_df['Deadline'].fillna('') + ' ' +
    filtered_df['Location'].fillna('') + ' ' +
    filtered_df['Duty'].fillna('')
)

    # TfidfVectorizer를 적용
    combined_matrix = vectorizer.fit_transform(filtered_df['combined_text'])

    # 사용자 질의를 벡터화
    query_vector = vectorizer.transform([query_preprocessed])

    # 코사인 유사도 계산
    similarity_scores = cosine_similarity(query_vector, combined_matrix).flatten()

    # 유사도 순으로 상위 5개의 인덱스 추출
    top_indices = similarity_scores.argsort()[-5:][::-1]

    # 상위 5개의 회사, 제목, URL을 추천
    recommendations = filtered_df[['Title', 'Company', 'URL']].iloc[top_indices]
    
    print("추천 결과:")
    print(recommendations)


# In[ ]:


# 채용 공고 추천 시스템 실행
recommend_interactively(df, okt)


# In[ ]:





# In[ ]:





#  
