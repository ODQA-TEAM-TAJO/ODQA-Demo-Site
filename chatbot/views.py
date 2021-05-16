from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

from elasticsearch import Elasticsearch

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')

josa = list(set(['은','는','을','를','이','가','의','에','로','으로','과','와','도','에서']))
seperation = ['\n', '\n\n','\\n','\\n\\n','\"']
special_tokens_dict = {'additional_special_tokens': josa+seperation}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

model = AutoModelForQuestionAnswering.from_pretrained('/Users/kimnamhyeok/PycharmProjects/djangoProject/checkpoint-2300')

model.resize_token_embeddings(len(tokenizer))

nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)

def home(request):
    context = {}

    return render(request, "chathome.html", context)

@csrf_exempt
def chatanswer(request):

    es = Elasticsearch('localhost:9200')

    context = {}

    questext = request.GET['questext']

    import colorama
    colorama.init()
    from colorama import Fore, Style

    def retrieval(question):

        query = {
            'query': {
                'bool': {
                    'should': [
                        {'match': {'title': question}},
                        {'match': {'text': question}}
                    ]
                }
            }
        }

        doc = es.search(index='document',body=query,size=10)['hits']['hits']

        return doc

    def mrc(question,document):

        for i in range(len(document)):
            answer = nlp(question=question, context=document[i]['_source']['text'])
            break

        print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL, answer['answer'])

        return answer['answer']

    if questext == '안녕하세요.':
        anstext = '안녕하세요.'
    elif questext == '누가 만들었어?':
        anstext = '부스트캠프 AI Tech MRC 타조가 만들었습니다.'
    else:
        anstext = mrc(questext,retrieval(questext))

    print(anstext)

    context['anstext'] = anstext
    context['flag'] = '0'

    context['result'] = 'sss'

    return JsonResponse(context, content_type="application/json")