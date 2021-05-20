import onnxruntime as ort
import torch
import torch.nn as nn
import re
import torch
import argparse
import numpy as np

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm 

import onnx
from onnxruntime.quantization import QuantizationMode, quantize_dynamic, quantize_qat, QuantType

def preprocess_text(text: str) -> str:
    """
    text cleaning
    :param text: str
    :return: str
    """
    text = text.lower()
    text = re.sub("[^а-яА-Яa-zA-Z0-9ё\-\\\@/+=_%№ ]", " ", text)
    text = re.sub(r"ё", "е", text)
    text = re.sub("\-\s+", " ", text)
    text = re.sub("\s+", " ", text)
    text = text.strip()
    return text


args = argparse.Namespace(
    config_name = "ru_conversational_cased_L-12_H-768_A-12_pt",
    tokenizer_name = "ru_conversational_cased_L-12_H-768_A-12_pt",
    model_name_or_path = "ru_conversational_cased_L-12_H-768_A-12_pt",
    num_labels = 3,
    device = "cuda", 
    maxlen = 128,
    export_model_path = "outdir/14_0.79_sentim.onnx"
)

config = AutoConfig.from_pretrained(
    args.config_name if args.config_name else args.model_name_or_path,
    num_labels=args.num_labels
)

tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
)

### Quantization
if True:
    onnx_model_path = 'model.onnx'
    quantized_model_path = f"model-quantized.onnx"

    quantized_model = quantize_dynamic(onnx_model_path, quantized_model_path)


# text = preprocess_text("здравствуйте скажите пожалуйста как мне отключить данные номер на время")
# inputs = tokenizer(text, max_length=args.maxlen, padding="max_length", return_tensors="np")
# input_ids = inputs['input_ids']
# token_type_ids = inputs['token_type_ids']
# attention_mask = inputs['attention_mask']

ort_session = ort.InferenceSession('model-quantized.onnx')


# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# ort_inputs = {
#     ort_session.get_inputs()[0].name: input_ids,
#     ort_session.get_inputs()[1].name: token_type_ids,
#     ort_session.get_inputs()[2].name: attention_mask,
# } 
 
# pred = ort_session.run(['output'], ort_inputs)[0]
# pred_output_softmax = np.argmax(pred)
# _, predicted = torch.max(pred_output_softmax, 1)

with open("data/sentiment_data/labels.txt") as l_file:
    labels = l_file.read().split("\n")

with open("data/sentiment_data/test_examples.txt") as s_file:
    test_samples = s_file.read().split("\n")

with open("data/sentiment_data/test_labels.txt") as tl_file:
    test_labels = tl_file.read().split("\n")

true_predictions_count = 0
total = 0
for i in tqdm(range(len(test_samples))):
    text = test_samples[i]
    label = test_labels[i] 

    inputs = tokenizer(text, max_length=args.maxlen, padding="max_length", return_tensors="np")
    ort_inputs = {
        ort_session.get_inputs()[0].name: inputs['input_ids'],
        ort_session.get_inputs()[1].name: inputs['token_type_ids'],
        ort_session.get_inputs()[2].name: inputs['attention_mask'],
    } 

    output = ort_session.run(['output'], ort_inputs)
    prediction = labels[np.argmax(output[0])]

    if prediction == label:
        true_predictions_count += 1
    
    total += 1

print(f"Accuracy {true_predictions_count/total*100:.2f}")