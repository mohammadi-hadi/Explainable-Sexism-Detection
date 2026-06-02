#!/usr/bin/env python3
"""
Multi-model comparison for sexism detection on the EXIST shared task.

Trains and compares four transformer configurations on EXIST Task 1
(binary: sexist / not sexist):
  * multilingual BERT
  * XLM-RoBERTa
  * DistilBERT
  * a concatenated ensemble of all three

Each configuration is tuned with a small Optuna hyper-parameter search and
trained on synonym / swap / char-insertion augmented data; results are reported
as accuracy, precision, recall, and macro-F1 on a held-out test split.

Companion code for Chapter 3 of the PhD thesis
"Let Me Explain! Explainable NLP for Understanding Large Language Models".

The EXIST 2023 data is obtained from the shared-task organisers (see
data/ethics_reference.md); set the paths below before running.
"""
import json
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw

import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)
from transformers import (
    BertTokenizer, BertModel, XLMRobertaModel, DistilBertModel,
)

nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("omw-1.4", quiet=True)

# --- Configuration -----------------------------------------------------------
EXIST_JSON = "EXIST2023_training.json"   # EXIST 2023 Task 1 training file
ADDITIONAL_CSV = "train_all_tasks.csv"   # additional English (EDOS) subset
SEED = 42
MAX_LENGTH = 256
NUM_EPOCHS = 5
N_TRIALS = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Text preprocessing & augmentation ---------------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"(https?://\S+|www\.\S+)", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return " ".join(lemmatizer.lemmatize(w) for w in words)


aug_synonym = naw.SynonymAug(aug_src="wordnet")
aug_swap = naw.RandomWordAug(action="swap")
aug_insert = nac.RandomCharAug(action="insert")


def augment_text(text, num_augmentations=3):
    augmented = []
    for _ in range(num_augmentations):
        t = text
        t = aug_synonym.augment(t)
        t = aug_swap.augment(t)
        t = aug_insert.augment(t)
        augmented.append(t)
    return augmented


def majority_voting(labels):
    num_yes = sum(label == "YES" for label in labels)
    num_no = sum(label == "NO" for label in labels)
    return "YES" if num_yes > num_no else "NO"


# --- Data loading -------------------------------------------------------------
def load_dataset():
    """Load the EXIST + additional English data, augment, and binarise labels."""
    with open(EXIST_JSON) as f:
        data = json.load(f)

    rows = []
    for key in data:
        if data[key]["labels_task1"] == []:
            continue
        rows.append({
            "id_EXIST": data[key]["id_EXIST"],
            "text": preprocess_text(data[key]["tweet"]),
            "label1": data[key]["labels_task1"],
            "lang": data[key]["lang"],
            "source": "original",
        })
    df = pd.DataFrame(rows)

    # Additional English subset (EDOS "train_all_tasks"); labels mapped to YES/NO
    add = pd.read_csv(ADDITIONAL_CSV).rename(
        columns={"rewire_id": "id_EXIST", "label_sexist": "label1"})
    add["label1"] = add["label1"].replace({"sexist": "YES", "not sexist": "NO"})
    add["text"] = add["text"].apply(preprocess_text)
    add["lang"] = "en"
    add["source"] = "additional"
    df = add[["id_EXIST", "text", "label1", "lang", "source"]]

    # Data augmentation
    augmented_rows = []
    for _, row in df.iterrows():
        for aug in augment_text(row["text"]):
            r = row.to_dict()
            r["text"] = aug
            augmented_rows.append(r)
    train_df = pd.DataFrame(augmented_rows).sample(
        frac=1, random_state=SEED).reset_index(drop=True)

    # Binary label via majority voting over the per-annotator labels
    def to_binary(lab):
        if isinstance(lab, list):
            return 1 if majority_voting(lab) == "YES" else 0
        return 1 if lab == "YES" else 0

    labels = [to_binary(lab) for lab in train_df["label1"]]
    return train_df, labels


class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# --- Model --------------------------------------------------------------------
class CustomBERTModel(nn.Module):
    """BERT / XLM-R / DistilBERT, individually or concatenated as an ensemble."""

    def __init__(self, num_labels, model_type="all"):
        super().__init__()
        self.model_type = model_type
        self.bert_model = self.xlm_roberta_model = self.distilbert_model = None
        if model_type in ("all", "bert"):
            self.bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")
        if model_type in ("all", "xlm_roberta"):
            self.xlm_roberta_model = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        if model_type in ("all", "distilbert"):
            self.distilbert_model = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")
        width = 768 * 3 if model_type == "all" else 768
        self.fc = nn.Linear(width, num_labels)

    def forward(self, input_ids):
        outputs = []
        if self.bert_model is not None:
            outputs.append(self.bert_model(input_ids).last_hidden_state[:, 0, :])
        if self.xlm_roberta_model is not None:
            outputs.append(self.xlm_roberta_model(input_ids).last_hidden_state[:, 0, :])
        if self.distilbert_model is not None:
            outputs.append(self.distilbert_model(input_ids).last_hidden_state[:, 0, :])
        feats = torch.cat(outputs, dim=1) if self.model_type == "all" else outputs[0]
        return self.fc(feats)


# --- Experiment ---------------------------------------------------------------
def run_experiment(model_type, train_dataset, test_dataset):
    """Optuna search on training loss, then train and evaluate on the test set."""

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        model = CustomBERTModel(num_labels=2, model_type=model_type).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        total_loss = 0.0
        for _ in range(NUM_EPOCHS):
            model.train()
            total_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()
                out = model(batch["input_ids"].to(device))
                loss = criterion(out, batch["labels"].to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        return total_loss / len(loader)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)
    best = study.best_params

    model = CustomBERTModel(num_labels=2, model_type=model_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=best["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(train_dataset, batch_size=best["batch_size"], shuffle=True)
    model.train()
    for _ in range(3):
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch["input_ids"].to(device))
            loss = criterion(out, batch["labels"].to(device))
            loss.backward()
            optimizer.step()

    test_loader = DataLoader(test_dataset, batch_size=best["batch_size"], shuffle=False)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch["input_ids"].to(device))
            _, pred = torch.max(out.data, 1)
            y_true.extend(batch["labels"].cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }


def main():
    train_df, labels = load_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(
        train_df, labels, test_size=0.2, stratify=labels, random_state=SEED)

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
    enc_tr = tokenizer(X_tr["text"].astype(str).tolist(), truncation=True,
                       padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
    enc_te = tokenizer(X_te["text"].astype(str).tolist(), truncation=True,
                       padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
    train_dataset = CustomDataset(enc_tr, y_tr)
    test_dataset = CustomDataset(enc_te, y_te)

    results = {}
    for model_type in ("bert", "xlm_roberta", "distilbert", "all"):
        print(f"Running experiment for {model_type}")
        results[model_type] = run_experiment(model_type, train_dataset, test_dataset)

    print(pd.DataFrame(results).T.round(3))


if __name__ == "__main__":
    main()
