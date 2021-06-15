from transformers import AutoTokenizer
import torch
import numpy as np
import onnxruntime as ort
from onnxruntime import SessionOptions, ExecutionMode


class MRC:
    def __init__(self, model_path, tokenizer_path, tag_predict_model):
        options = SessionOptions()
        options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
        self.model = ort.InferenceSession(model_path, options)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.tag_model = tag_predict_model

    def get_answer(self, context, question):
        tag = self.tag_model.get_tag(question)
        # print(tag)

        question_input  = tag + question
        context_input  = context

        tokenized_examples = self.tokenizer(
                question_input,
                context_input+context_input,
                truncation="only_second",
                max_length=512,
                padding="max_length",
                stride = 128,
                return_overflowing_tokens=True
            )

        all_answer = []
        all_score = []


        # QUESTION+CONTEXT 가 긴 경우 여러번 나눠서 결과 추론. 
        for input_ids, attention_mask, token_type_ids in zip(tokenized_examples["input_ids"],tokenized_examples["attention_mask"],tokenized_examples["token_type_ids"]):    
            onnx_input = {
                "input_ids": np.array([input_ids]),
                "attention_mask": np.array([attention_mask]),
                "token_type_ids": np.array([token_type_ids])
            }
            outputs = self.model.run(None, onnx_input)

            answer_start_scores = torch.tensor(outputs[0])
            answer_end_scores = torch.tensor(outputs[1])
                
            answer_start = torch.topk(answer_start_scores, k =2)
            answer_end = torch.topk(answer_end_scores, k =2)
            
            for score_start, idx_start in zip(answer_start.values.tolist()[0], answer_start.indices.tolist()[0]):
                for score_end, idx_end in zip(answer_end.values.tolist()[0], answer_end.indices.tolist()[0]):
                    if idx_start<idx_end+1:
                        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[idx_start:idx_end+1]))
                        all_answer.append(answer) 
                        all_score.append(score_start+score_end) 

        # 점수가 가장 높은 answer 한개 리턴
        if len(all_answer)!=0:
            answer =all_answer[np.argmax(all_score)]
        else:
            answer = 'no answer'
        return answer
