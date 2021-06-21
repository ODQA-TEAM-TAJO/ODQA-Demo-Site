import numpy as np
from tqdm import tqdm
import pickle
from haystack.retriever.dense import DensePassageRetriever
from haystack.document_store.faiss import FAISSDocumentStore

import onnxruntime as ort
from transformers import AutoTokenizer

class DPR:
    def __init__(self, model_path, tokenizer_path, document_store):
        self.question_encoder = ort.InferenceSession(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.document_store = document_store
        print('DPR setting completed')

    def get_documents(self, query, top_k):
        inputs = self.tokenizer.encode_plus(query)
        tokens = {name: np.atleast_2d(value) for name, value in inputs.items()}
        output = self.question_encoder.run(None, tokens)

        documents = self.document_store.query_by_embedding(output[0], top_k=top_k)
        return documents