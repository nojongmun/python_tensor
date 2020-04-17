from nltk.tokenize import word_tokenize
from konlpy.tag import Okt
import pandas as pd
from nltk import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# 자연어 처리 (위치) ; https://konlpy-ko.readthedocs.io/ko/v0.4.3/ ( 한국어 nlp 설치(pip install konlpy))
# 설치 ;
# pip install konlpy
# pip install nltk
# pip install wordcloud
class Service:
    def __init__(self):
        self.texts = []      # 텏스트 모임
        self.tokens = []     # 토큰 모임
        self.stopwords = []
        self.freqtxt = []
        self.okt = Okt()

    def extract_tokens(self,payload):
        print('텍스트 문서에서 token 추출')
        filename = payload.context + payload.fname
        with open(filename,'r', encoding='utf-8') as f:
            self.texts = f.read()
        # print(f'{self.texts[:300]}')

    def extract_hangeul(self):
        print('한글 추출')
        texts = self.texts.replace('\n', ' ')
        tokenizer = re.compile(r'[^ ㄱ-힣 ]')
        self.texts = tokenizer.sub('', texts)
        # print(f'{self.texts[:300]}')

    def conversion_token(self):
        print('토큰으로 변환')
        self.tokens = word_tokenize(self.texts)
        # print(f'{self.tokens[:300]}')

    # 복합명사 처리(조사 처리)
    def compound_noun(self):
        print('복합명사는 묶어서 filtering 으로 출력')
        print('예: 삼성전자의 스마트폰은  ---> 삼성전자  스마트폰')
        noun_tokens = []
        for token in self.tokens:
            token_pos = self.okt.pos(token)
            temp = [txt_tag[0] for txt_tag in token_pos
                    if txt_tag[1] == 'Noun']
            if len("".join(temp)) > 1:
                noun_tokens.append("".join("".join(temp)))
        self.texts = " ".join(noun_tokens)
        # print(f'{self.texts[:300]}')

    def extract_stopword(self, payload):
        print('스톱워드 추출')
        filename = payload.context + payload.fname
        with open(filename, 'r', encoding='utf-8') as f:
            self.stopwords = f.read()
        self.stopwords = self.stopwords.split(' ')


    def filtering_text_with_stopword(self):
        print('스톱워드 필터링')
        self.texts = word_tokenize(self.texts)
        self.texts = [text for text in self.texts
                      if text not in self.stopwords]

        # print(f'{self.texts[:300]}')

    # 빈도수 추출
    def frequent_text(self):
        print('빈도수 정렬')
        # 많이 사용 되는 자료가 위로 (내림차순)
        self.freqtxt = pd.Series(dict(FreqDist(self.texts))).sort_values(ascending=False)
        print(f'{self.freqtxt[:300]}')


    def draw_worldcloud(self,payload):
        print('워드 크라우드 생성')
        filename = payload.context + payload.fname
        wcloud = WordCloud(filename, relative_scaling=0.2, background_color='white').generate(" ".join(self.texts))

        plt.figure(figsize=(12,12))
        plt.imshow(wcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()











