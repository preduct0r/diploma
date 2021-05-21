import re
import json
import torch
import argparse
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torch import optim
from sklearn.metrics import accuracy_score
import numpy as np
import scipy
from tqdm import tqdm
from telegram_bot import telegram_bot_sendtext
import pandas as pd


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

def calculate_statistics(samples, confidence_level=0.95):
    min_, max_ = min(samples), max(samples)
    mean = np.mean(samples)
    sstddev = np.std(samples, ddof=1)  # sample stddev
    stderr = sstddev / np.sqrt(len(samples))
    alpha = (1 - confidence_level) / 2
    radius = scipy.stats.t.ppf(1 - alpha, len(samples) - 1) * stderr
    return min_, max_, mean, radius, sstddev

args = argparse.Namespace(
    config_name = "/home/den/MODELS/ru_conversational_cased_L-12_H-768_A-12",
    tokenizer_name = "/home/den/MODELS/ru_conversational_cased_L-12_H-768_A-12",
    model_name_or_path = "/home/den/MODELS/ru_conversational_cased_L-12_H-768_A-12",
    test_data = "/home/den/PycharmProjects/diploma/text/asr_data/ramaz_asr_simple.csv",
    cache_dir = "cache_dir",
    device = "cuda",
    epochs = 1,
    lr = 2e-5,
    maxlen = 128,
    clip = 0.15,
    warmup_steps = 50
)


class ClassierDataset(Dataset):
    def __init__(self, data_path, tokenizer, device="cuda", maxlen=128):
        self.tokenizer = tokenizer
        self.device = device
        self._maxlen = maxlen
        self._data = self._read_data(data_path)

    def _read_data(self, data_path):
        data = []
        src_data = pd.read_csv(open(data_path))

        for item in src_data['chunk']:
            sample = tokenizer.encode(preprocess_text(item))

            if len(sample) < self._maxlen:
                sample += (self._maxlen - len(sample)) * tokenizer.encode("[PAD]", add_special_tokens=False)
            else:
                sample = sample[:self._maxlen]

            data += [{"chunk": item, "x": sample}]

        return data

    def __getitem__(self, index):
        sample = self._data[index]

        return {"chunk": sample["chunk"], "x": torch.tensor(sample["x"]).to(self.device)}

    def __len__(self):
        return len(self._data)


tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
)

test_dataset = ClassierDataset(args.test_data, tokenizer, device=args.device, maxlen = args.maxlen)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#TODO хардкод, печально, но как без него(для тональности плевать впринципе)
num_labels = 2

config = AutoConfig.from_pretrained(
    args.config_name if args.config_name else args.model_name_or_path,
    num_labels=num_labels
)

best_model_path = "/home/den/PycharmProjects/diploma/text/1_0.8945538818076477_sentim.pt"

model = AutoModelForSequenceClassification.from_pretrained(
    best_model_path,
    config=config
)

model = model.to(args.device)

model.eval()
preds, chunks = [], []

for batch in tqdm(test_dataloader, desc='Evaluating'):
    output = model(batch["x"])
    preds.append(output['logits'].detach().cpu().numpy().squeeze().tolist())
    chunks += batch["chunk"]

temp_df = pd.DataFrame(columns=["chunk", "pos", "neg"])
temp_df["chunk"] = chunks
temp_df.loc[:,["pos", "neg"]] = np.vstack(preds)

asr_simple = pd.read_csv(open(args.test_data))

merge_df = pd.merge(asr_simple, temp_df, on='cur_name', how='outer')
merge_df.to_csv('/home/den/Documents/diploma/asr_yandex/bert_inference.csv', index=False)





