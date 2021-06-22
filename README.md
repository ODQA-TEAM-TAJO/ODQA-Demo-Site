# ODQA-Demo-Site v2.1

<p align="center"><img src="https://user-images.githubusercontent.com/55614265/122911158-f5a6cc80-d391-11eb-8e15-6bed5c6bac1c.gif"/></p>

## Architecture

<img src="https://user-images.githubusercontent.com/55614265/122911276-17a04f00-d392-11eb-8720-98616eb11989.png" height="751px" width="458px"/>

- Guideline : Explain how to use it for 20 seconds for the first user.
- Preprocess question : After the question is entered, correct the spelling.
- Daily chat : Rule-based daily conversation
- Short question : For short questions of less than two syllables, it answers wiki url by extracting keywords from those questions.
- Expected Q&A : If there is a highly similar expected question, output the answer to the expected question.
- Retrieval : If it is impossible to retrieve a document, reply that "I don't understand your question.".
- Reader : If the MRC's answer does not exceed the standard score, chatbot says "I'm not sure.".
- Postprocess answer : In the case of Korean, there is an investigation, so it is removed.

## Usage
```
$> tree -d
.
├── static
│     ├── css
│     ├── image
│     └── js
├── templates
├── app.py
├── dpr_inference.py
├── mrc_inference.py
├── tag_inference.py
├── Retrieval.ipynb
└── Setting.ipynb
```

1. Setting.ipynb : Install the required libraries.
2. Retrieval.ipynb : Set Elasticsearch.
3. app.py : Load the required modules, then run flask.
