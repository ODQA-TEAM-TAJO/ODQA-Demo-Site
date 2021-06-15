from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
import onnxruntime as ort
from onnxruntime import SessionOptions, ExecutionMode


label_decoder = {0: '[WHO]', 1: '[WHEN]', 2: '[WHERE]', 3: '[WHAT]', 4: '[HOW]', 5: '[WHY]', 6: '[QUANTITY]', 7: '[CITE]'}

class TagInference:
    def __init__(self, model_path, tokenizer_path):
        options = SessionOptions()
        options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
        self.model = ort.InferenceSession(model_path, options)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    def get_tag(self, question):
        tokenized_sent = self.tokenizer(question, return_tensors='pt')
        onnx_input = {
                "input_ids": np.array(tokenized_sent['input_ids']),
                "attention_mask": np.array(tokenized_sent['attention_mask']),
                "token_type_ids": np.array(tokenized_sent['token_type_ids'])
        }
        outputs = self.model.run(None, onnx_input)
        pred = np.argmax(outputs[0], axis=-1)
        return label_decoder[pred[0]]


