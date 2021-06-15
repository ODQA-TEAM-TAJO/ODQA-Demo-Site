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

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BertForTokenClassification, pipeline
import onnxruntime as ort
import numpy as np
import torch

tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-finetuned-korquad")

model = ort.InferenceSession('/home/dr_lunars/electra_reader_small-optimized-quantized.onnx')


def mrc(context, question):
    tag = '[WHEN]'

    question_input = tag + question
    context_input = context

    tokenized_examples = tokenizer(
        question_input,
        context_input + context_input,
        truncation="only_second",
        max_length=512,
        padding="max_length",
        stride=128,
        return_overflowing_tokens=True
    )

    all_answer = []
    all_score = []

    for input_ids, attention_mask, token_type_ids in zip(tokenized_examples["input_ids"],
                                                         tokenized_examples["attention_mask"],
                                                         tokenized_examples["token_type_ids"]):
        onnx_input = {
            "input_ids": np.array([input_ids]),
            "attention_mask": np.array([attention_mask]),
            "token_type_ids": np.array([token_type_ids])
        }
        outputs = model.run(None, onnx_input)

        answer_start_scores = torch.tensor(outputs[0])
        answer_end_scores = torch.tensor(outputs[1])

        answer_start = torch.topk(answer_start_scores, k=2)
        answer_end = torch.topk(answer_end_scores, k=2)

        for score_start, idx_start in zip(answer_start.values.tolist()[0], answer_start.indices.tolist()[0]):
            for score_end, idx_end in zip(answer_end.values.tolist()[0], answer_end.indices.tolist()[0]):
                if idx_start < idx_end + 1:
                    answer = tokenizer.convert_tokens_to_string(
                        tokenizer.convert_ids_to_tokens(input_ids[idx_start:idx_end + 1]))
                    all_answer.append(answer)
                    all_score.append(score_start + score_end)

    if len(all_answer) != 0:
        answer = all_answer[np.argmax(all_score)]
    else:
        answer = 'no answer'
    return answer, max(all_score)

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
                ans = mrc(doc[i]['_source']['text'], q)
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