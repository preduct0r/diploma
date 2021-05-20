import re
import torch
import argparse

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm 

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

model = AutoModelForSequenceClassification.from_config(config)

model = model.to(args.device)
model.load_state_dict(torch.load("outdir/14_0.79_sentim.pt")["model_state_dict"])
model.eval()

'''
Define arguments to pass to onnx exporter
'''
model_onnx_path = "model_cf.onnx"

text = preprocess_text("здравствуйте скажите пожалуйста как мне отключить данные номер на время")
inputs = tokenizer(text, max_length=args.maxlen, padding="max_length")
input_ids = torch.tensor(inputs['input_ids']).unsqueeze(0).to(args.device)
token_type_ids = torch.tensor(inputs['token_type_ids']).unsqueeze(0).to(args.device)
attention_mask = torch.tensor(inputs['attention_mask']).unsqueeze(0).to(args.device)

# The inputs "input_ids", "token_type_ids" and "attention_mask" are torch tensors of shape batch*seq_len
dummy_input = (input_ids, token_type_ids, attention_mask)
input_names = ["input_ids", "token_type_ids", "attention_mask"]
output_names = ["output"]

'''
convert model to onnx
'''
dynamic_axes = {'input_ids': {0: 'batch'}, 'token_type_ids': {0: 'batch'}}
torch.onnx.export(
    model, 
    dummy_input, 
    model_onnx_path,
    opset_version=11,  
    input_names = input_names, 
    output_names = output_names,
    do_constant_folding=True,
    dynamic_axes={'input_ids' : {0 : 'batch_size', 1: 'seq_len'}, 'token_type_ids' : {0 : 'batch_size', 1: 'seq_len'}, 'attention_mask' : {0 : 'batch_size', 1: 'seq_len'}, 'output' : {0 : 'batch_size', 1: 'num_labels'}}, verbose=False)
