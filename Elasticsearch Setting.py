# !pip install elasticsearch # 엘라스틱 서치를 파이썬에서 사용하기 위한 라이브러리 설치
# elasticsearch-7.12.1/bin/elasticsearch-plugin install analysis-nori # 한국어 토크나이저 설치
# elasticsearch-7.12.1/bin/elasticsearch 실행

import json
import re
import kss
import pandas as pd
from tqdm import tqdm
from elasticsearch import Elasticsearch

es = Elasticsearch('localhost:9200')

es.indices.create(index = 'document',
                  body = {
                      'settings':{
                          'analysis':{
                              'analyzer':{
                                  'my_analyzer':{
                                      "type": "custom",
                                      'tokenizer':'nori_tokenizer',
                                      'decompound_mode':'mixed',
                                      'stopwords':'_korean_',
                                      'synonyms':'_korean_',
                                      "filter": ["lowercase",
                                                 "my_shingle_f",
                                                 "nori_readingform",
                                                 "nori_number",
                                                 "cjk_bigram",
                                                 "decimal_digit",
                                                 "stemmer",
                                                 "trim"]
                                  }
                              },
                              'filter':{
                                  'my_shingle_f':{
                                      "type": "shingle"
                                  }
                              }
                          },
                          'similarity':{
                              'my_similarity':{
                                  'type':'BM25',
                              }
                          }
                      },
                      'mappings':{
                          'properties':{
                              'title':{
                                  'type':'text',
                                  'analyzer':'my_analyzer',
                                  'similarity':'my_similarity'
                              },
                              'text':{
                                  'type':'text',
                                  'analyzer':'my_analyzer',
                                  'similarity':'my_similarity'
                              }
                          }
                      }
                  }
                  )

with open('/Users/kimnamhyeok/PycharmProjects/djangoProject/wikipedia_documents.json', 'r') as f:
    wiki_data = pd.DataFrame(json.load(f)).transpose()

wiki_data = wiki_data.drop_duplicates(['text'])

wiki_data = wiki_data.reset_index()

del wiki_data['index']

wiki_data['text_origin'] = wiki_data['text']

wiki_data['text'] = wiki_data['text'].apply(lambda x : x.replace('\\n\\n',' '))
wiki_data['text'] = wiki_data['text'].apply(lambda x : x.replace('\n\n',' '))
wiki_data['text'] = wiki_data['text'].apply(lambda x : x.replace('\\n',' '))
wiki_data['text'] = wiki_data['text'].apply(lambda x : x.replace('\n',' '))

wiki_data['text'] = wiki_data['text'].apply(lambda x : ' '.join(re.sub(r'[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣.,!?]', ' ', str(x.lower().strip())).split()))

title = []
text = []

for num in tqdm(range(len(wiki_data))):
    try:
        for doc in kss.split_chunks(wiki_data['text'][num], max_length=1024, overlap=False):
            title.append(wiki_data['title'][num])
            text.append(doc[1])
    except:
        title.append(wiki_data['title'][num])
        text.append(wiki_data['text'][num])

df = pd.DataFrame({'title':title,'text':text})

for num in tqdm(range(len(df))):
    es.index(index='document', body = {"title" : df['title'][num], "text" : df['text'][num]})