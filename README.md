# ODQA-Demo-Site v2.0

<p align="center"><img src="https://user-images.githubusercontent.com/55614265/121783568-4e37d600-cbea-11eb-8ec3-486442762e8b.gif"/></p>

## Architecture

<img src="https://user-images.githubusercontent.com/55614265/121783008-7b36b980-cbe7-11eb-8d2f-9b609e501a4f.png" height="552px" width="343px"/>

- Preprocess question : After the question is entered, correct the spelling.
- Expected Q&A : If there is a highly similar expected question, output the answer to the expected question.
- Retrieval : If it is impossible to retrieve a document, reply that "I don't understand your question.".
- MRC : If the MRC's answer does not exceed the standard score, chatbot says "I'm not sure.".
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
├── Demo_Site.ipynb
├── Retrieval.ipynb
└── Setting.ipynb
```

1. Setting.ipynb : Install the required libraries.
2. Retrieval.ipynb : Set Elasticsearch.
3. Demo_Site.ipynb : Load the required modules, then run flask.
