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
    train_data = "/home/den/DATASETS/TEXT/fns_and_beeline_stratified/20cl_train_stratified.json",
    test_data = "/home/den/DATASETS/TEXT/fns_and_beeline_stratified/20cl_test_stratified.json",
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
        self._labels = []
        self._maxlen = maxlen
        self._data = self._read_data(data_path)

    def _read_data(self, data_path):
        data = []
        src_data = json.load(open(data_path))

        for item in src_data:
            self._labels += [item["classId"]]
            for sample in item["samples"]:
                sample = tokenizer.encode(preprocess_text(sample))

                if len(sample) < self._maxlen:
                    sample += (self._maxlen - len(sample)) * tokenizer.encode("[PAD]", add_special_tokens=False)
                else:
                    sample = sample[:self._maxlen]

                data += [{"y": self._labels.index(item["classId"]), "x": sample}]

        return data

    def __getitem__(self, index):
        item = self._data[index]

        return {"y": torch.tensor(item["y"]).to(self.device), "x": torch.tensor(item["x"]).to(self.device)}

    def __len__(self):
        return len(self._data)

tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
)

dataset = ClassierDataset(args.train_data, tokenizer, device=args.device, maxlen = args.maxlen)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)



test_dataset = ClassierDataset(args.test_data, tokenizer, device=args.device, maxlen = args.maxlen)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)



num_labels = len(dataset._labels)

# test_data = torch.tensor(test_data)

config = AutoConfig.from_pretrained(
    args.config_name if args.config_name else args.model_name_or_path,
    num_labels=num_labels
)

model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name_or_path,
    from_tf=bool(".ckpt" in args.model_name_or_path),
    config=config
)

model = model.to(args.device)


optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=len(dataset)
)

best_acc = 0
best_epoch = 0
for epoch in range(args.epochs):
    model.train()
    for batch in tqdm(dataloader, desc='Training'):
        optimizer.zero_grad()

        loss, logits = model(input_ids=batch["x"], labels=batch["y"], return_dict=False)
        loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        scheduler.step()

    model.eval()
    preds, targets = [], []

    for batch in tqdm(test_dataloader, desc='Evaluating'):
        output = model(batch["x"])
        preds += [output[0][0].argmax(0).item()]
        targets += batch["y"].detach().cpu()
        
    acc = accuracy_score(targets, preds)
    print(f"Current accuracy: {acc}")

    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch

        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'acc': acc
        # }, f"outdir/{epoch}_{acc}_sentim.pt")

print(f"Best Epoch: {best_epoch} Best Acc: {best_acc}")
telegram_bot_sendtext("Обучение берта закончилось!")

