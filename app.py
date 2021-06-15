# %%
# Retrieval

from subprocess import Popen, PIPE, STDOUT
import time
from elasticsearch import Elasticsearch

es_server = Popen(['/home/dr_lunars/elasticsearch-7.0.0/bin/elasticsearch'],stdout=PIPE, stderr=STDOUT)

time.sleep(30)

es = Elasticsearch("http://localhost:9200", timeout=300, max_retries=10, retry_on_timeout=True)

# %%
# Reader
tag_model_path = '/content/drive/MyDrive/haystack_tutorial/Reader/tag-optimized-quantized.onnx'
tag_tokenizer_path = '/content/drive/MyDrive/haystack_tutorial/Reader/no_compounds/bert/checkpoint-1500'

mrc_model_path = '/content/drive/MyDrive/haystack_tutorial/Reader/reader_small-optimized-quantized.onnx'
mrc_tokenizer_path = '/content/drive/MyDrive/haystack_tutorial/Reader/no_compounds/koelectra_small/checkpoint-2900'

from mrc_inference import MRC
from tag_inference import TagInference

tag_model = TagInference(
    model_path=tag_model_path,
    tokenizer_path=tag_tokenizer_path
)
mrc = MRC(
    model_path=mrc_model_path,
    tokenizer_path=mrc_tokenizer_path,
    tag_predict_model=tag_model
)
# %%
# NER

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

if not os.path.isfile('log.txt'):
    f = open("log.txt", 'a+')
    f.write('question, answer\n')
    f.close()

# %%
# Flask

from flask import Flask, render_template, request

app = Flask(__name__,static_folder='/home/dr_lunars/ODQA-Demo-Site/static',template_folder='/home/dr_lunars/ODQA-Demo-Site/templates')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    question = preprocess(request.args.get('msg'))
    questions = set([question,myInko.en2ko(question)])

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
        if ans_lst[0][1] >= 20:
            answer = ans_lst[0][0] + ' 입니다.'
            f = open("log.txt", 'a+')
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
        doc = es.search(index='document',body=query,size=5)['hits']['hits']
        if doc != []:
            max_scr = doc[0]['_score']
            for i in range(len(doc)):
                ans = mrc.get_answer(doc[i]['_source']['text'], q)
                ans_lst.append((ans[0],ans[1]*doc[i]['_score']/max_scr))
    if ans_lst != []:
        ans_lst = sorted(ans_lst, key = lambda x : x[1], reverse=True)
        if ans_lst[0][1] >= 0.7:
            answer = postprocess(ans_lst[0][0]) + ' 입니다.'
        else:
            answer = '잘 모르겠어요...'
    else:
        answer = '질문을 이해하지 못했어요...'
    f = open("log.txt", 'a+')
    f.write(question+', '+answer+'\n')
    f.close()
    return answer

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)