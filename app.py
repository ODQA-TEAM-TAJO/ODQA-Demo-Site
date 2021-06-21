# %%
# Retrieval

from subprocess import Popen, PIPE, STDOUT
import time
from elasticsearch import Elasticsearch

es_server = Popen(['/home/dr_lunars/elasticsearch-7.0.0/bin/elasticsearch'],stdout=PIPE, stderr=STDOUT)

time.sleep(30)

es = Elasticsearch("http://localhost:9200", timeout=300, max_retries=10, retry_on_timeout=True)

daily_score = 0

# %%
# DPR

# from haystack.document_store.faiss import FAISSDocumentStore
# document_store = FAISSDocumentStore.load(faiss_file_path="my_faiss", sql_url="sqlite:///my_doc_store.db", index="document")

# from dpr_inference import DPR

# model_path = '/home/dr_lunars/models/question_encoder-optimized-quantized.onnx'
# tokenizer_path = "kykim/bert-kor-base"

# dpr = DPR(
#     model_path=model_path,
#     tokenizer_path=tokenizer_path,
#     document_store=document_store
# )

# %%
# Reader

# tag_model_path = '/home/dr_lunars/models/tag-optimized-quantized.onnx'
# tag_tokenizer_path = '/home/dr_lunars/models/tokenizers/tag_bert'

# mrc_model_path = '/home/dr_lunars/models/electra_reader_small-optimized-quantized.onnx'
# mrc_tokenizer_path = '/home/dr_lunars/models/tokenizers/koelectra_small'

# from mrc_inference import MRC
# from tag_inference import TagInference

# tag_model = TagInference(
#     model_path=tag_model_path,
#     tokenizer_path=tag_tokenizer_path
# )
# mrc = MRC(
#     model_path=mrc_model_path,
#     tokenizer_path=mrc_tokenizer_path,
#     tag_predict_model=tag_model
# )

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BertForTokenClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained('monologg/koelectra-small-v3-finetuned-korquad')
model = AutoModelForQuestionAnswering.from_pretrained('monologg/koelectra-small-v3-finetuned-korquad')

mrc = pipeline("question-answering", model=model, tokenizer=tokenizer)

# %%
# Rerank

def rerank(sparse_documents, dense_documents=None):
    sparse_dict, dense_dict = {}, {}
    for sparse_doc, dense_doc in zip(sparse_documents, dense_documents):
        sparse_dict[sparse_doc['_source']['text']] = [sparse_doc['_score'], sparse_doc['_source']['title']]
        dense_dict[dense_doc.text] = dense_doc.score*0.1
    
    hybrid_docs = []
    for sparse_text, sparse_score_title in sparse_dict.items():
        hybrid_dict = {}
        hybrid_dict['_source'] = {
            'title': sparse_score_title[1],
            'text': sparse_text
        }
        try:
            hybrid_dict['_score'] = dense_dict[sparse_text] + sparse_score_title[0]
            hybrid_docs.append(hybrid_dict)
        except:
            hybrid_dict['_score'] =  sparse_score_title[0]
            hybrid_docs.append(hybrid_dict)
    
    hybrid_docs = sorted(hybrid_docs, key=lambda x: x['_score'], reverse=True)
    return hybrid_docs

# %%
# Preprocess

from hanspell import spell_checker
from inko import Inko

def preprocess(question):
    question = spell_checker.check(question).as_dict()['checked']
    return question

myInko = Inko(allowDoubleConsonant=False)

# %%
# Postprocess

from konlpy.tag import Hannanum
from konlpy.tag import Kkma
from konlpy.tag import Komoran
from konlpy.tag import Okt

hannanum = Hannanum()
kkma = Kkma()
komoran = Komoran()
okt = Okt()

def postprocess(ans):
    if hannanum.pos(ans)[-1][-1] in ['J']:
        ans = ans[:-len(hannanum.pos(ans)[-1][0])]
    elif kkma.pos(ans)[-1][-1] in ['JKS','JKC','JKG','JKO','JKM','JKI','JKQ','JC','JX']:
        ans = ans[:-len(kkma.pos(ans)[-1][0])]
    elif komoran.pos(ans)[-1][-1] in ['JKS','JKC','JKG','JKO','JKB','JKV','JKQ','JC','JX']:
        ans = ans[:-len(komoran.pos(ans)[-1][0])]
    elif okt.pos(ans)[-1][-1] in ['Josa']:
        ans = ans[:-len(okt.pos(ans)[-1][0])]
    return ans

postprocess('Loading...')

# %%
# Log

import os

if not os.path.isfile('Log.txt'):
    f = open("Log.txt", 'a+')
    f.write('question, answer\n')
    f.close()

# %%
# Answer

import random

Answer = ['잘 모르겠어요...','정확한 답변을 찾지 못했어요...','조금 더 구체적으로 질문해주세요...']
    
# %%
# Flask

from flask import Flask, render_template, request

app = Flask(__name__,static_folder='/home/dr_lunars/ODQA-Demo-Site/static',template_folder='/home/dr_lunars/ODQA-Demo-Site/templates')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    try:
        question = preprocess(request.args.get('msg'))
        questions = set([question,myInko.en2ko(question)])

        global daily_score

        ans_lst = []
        for q in questions:
            query = {
                'query':{
                    'bool':{
                        'must':[
                                {'match':{'question':q}}
                        ]
                    }
                }
            }
            doc = es.search(index='chatter',body=query,size=1)['hits']['hits']
            if doc != []:
                doc = doc[0]
                ans_lst.append((doc['_source']['answer'],doc['_score']))
        if ans_lst != []:
            ans_lst = sorted(ans_lst, key = lambda x : x[1], reverse=True)
            if ans_lst[0][1] >= 4:
                if daily_score == 3:
                    daily_score = 0
                    answer = '타조 챗봇은 간단한 일상 대화만 가능합니다. WIKI에서 찾을 수 있는 내용으로 질문해주세요.'
                    f = open("Log.txt", 'a+')
                    f.write(question+', '+answer+'\n')
                    f.close()
                    return answer
                else:
                    daily_score += 1
                    answer = ans_lst[0][0]
                    f = open("Log.txt", 'a+')
                    f.write(question+', '+answer+'\n')
                    f.close()
                    return answer

        daily_score = 0

        ans_lst = []
        for q in questions:
            q = q.replace("?","")
            if len(q.split()) == 1 or len(q.split()) == 2:
                query = {
                    'query':{
                        'bool':{
                            'must':[
                                    {'match':{'title': postprocess(q.split()[0])}}
                            ]
                        }
                    }
                }
                doc = es.search(index='document',body=query,size=1)['hits']['hits']
                if doc != []:
                    doc = doc[0]
                    ans_lst.append((doc['_source']['title'],doc['_score']))
        if ans_lst != []:
            ans_lst = sorted(ans_lst, key = lambda x : x[1], reverse=True)
            answer = '질문이 너무 짧아 정확한 답변을 하기 어렵습니다. <a href="https://ko.wikipedia.org/wiki/' + ans_lst[0][0] + '"  target="_blank">' + ans_lst[0][0] + '</a>을(를) 참고하세요.'

            f = open("Log.txt", 'a+')
            f.write(question+', '+answer+'\n')
            f.close()
            return answer
        
        if len(q.split()) <= 2:
            answer = '질문이 너무 짧아 정확한 답변을 하기 어렵습니다.'
            f = open("Log.txt", 'a+')
            f.write(question+', '+answer+'\n')
            f.close()
            return answer
            
        ans_lst = []
        for q in questions:
            query = {
                'query':{
                    'bool':{
                        'must':[
                                {'match':{'question':q}}
                        ]
                    }
                }
            }
            doc = es.search(index='qa',body=query,size=1)['hits']['hits']
            if doc != []:
                doc = doc[0]
                ans_lst.append((doc['_source']['answer'],doc['_score']))
        if ans_lst != []:
            ans_lst = sorted(ans_lst, key = lambda x : x[1], reverse=True)
            if ans_lst[0][1] >= 23:
                answer = ans_lst[0][0] + ' 입니다.'
                f = open("Log.txt", 'a+')
                f.write(question+', '+answer+'\n')
                f.close()
                return answer

        ans_lst = []
        for q in questions:
            query = {
                'query':{
                    'bool':{
                        'must':[
                                {'match':{'text':q}}
                        ]
                    }
                }
            }
            # sparse_docs = es.search(index='document',body=query,size=10)['hits']['hits']
            # dense_docs = dpr.get_documents(q, top_k=5)
            # doc = rerank(sparse_docs, dense_docs)
            doc = es.search(index='document',body=query,size=5)['hits']['hits']
            if doc != []:
                max_scr = doc[0]['_score']
                for i in range(len(doc)):
                    # ans = mrc.get_answer(context=doc[i]['_source']['text'], question=q)
                    ans = mrc(question=q, context=doc[i]['_source']['text'], topk=1) # tmp
                    # ans_lst.append((ans[0],ans[1]*doc[i]['_score']/max_scr))
                    ans_lst.append((ans['answer'],ans['score']*doc[i]['_score']/max_scr)) # tmp
        if ans_lst != []:
            ans_lst = sorted(ans_lst, key = lambda x : x[1], reverse=True)
            if ans_lst[0][1] >= 0.5:
                answer = postprocess(ans_lst[0][0]) + ' 입니다.'

            else:
                answer = Answer[random.randint(0,len(Answer)-1)]
        else:
            answer = '질문을 이해하지 못했어요...'
        f = open("Log.txt", 'a+')
        f.write(question+', '+answer+'\n')
        f.close()
        return answer
    except:
        answer = Answer[random.randint(0,len(Answer)-1)]
        f = open("Log.txt", 'a+')
        f.write(question+', '+answer+'\n')
        f.close()
        return answer

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)