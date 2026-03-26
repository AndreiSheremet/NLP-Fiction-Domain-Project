from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, AutoConfig, set_seed)
import torch
from torch.utils.data import DataLoader
import random
import evaluate
from tqdm.auto import tqdm

# Set random seeds
set_seed(42)

# Define hyperparameters (e.g., learning_rate, num_train_epochs, model_name)
learning_rate = 2e-5
num_train_epochs = 3
model_name = "google-bert/bert-base-cased"

def read_iob2(path):
    sentences = []
    labels = []

    cur_words = []
    cur_labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # New sentence starts
            if line.startswith("# sent_id"):
                if cur_words:
                    sentences.append(cur_words)
                    labels.append(cur_labels)
                    cur_words = []
                    cur_labels = []
                continue

            # Skip metadata
            if line.startswith("#"):
                continue

            # Skip empty lines
            if not line:
                continue

            parts = line.split()

            # Safety check
            if len(parts) < 3:
                continue

            word = parts[1]
            label = parts[2]

            cur_words.append(word)
            cur_labels.append(label)

    # last sentence
    if cur_words:
        sentences.append(cur_words)
        labels.append(cur_labels)

    return sentences, labels

train_sentences, train_labels =read_iob2('project/en_ewt-ud-train.iob2')
dev_sentences, dev_labels= read_iob2("project/en_ewt-ud-dev.iob2")
test_sentences, test_labels= read_iob2("project/en_ewt-ud-test-masked.iob2")

unique_labels = sorted({tag for sent_tags in train_labels for tag in sent_tags})
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}


# Load the tokenizer and model config
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
config = AutoConfig.from_pretrained(model_name, num_labels=len(label2id))

def align_labels(labels, word_ids, label2id):
    aligned = []
    prev_word_id = None

    for word_id in word_ids:
        if word_id is None:
            aligned.append(-100)
        elif word_id != prev_word_id:
            aligned.append(label2id[labels[word_id]])
        else:
            aligned.append(-100)
        prev_word_id = word_id

    return aligned

def tokenize_and_align_batch(sentences, labels, tokenizer, label2id):
    tokenized = tokenizer(
        sentences,
        is_split_into_words=True,
        truncation=True,
        padding=True
    )

    all_aligned_labels = []

    for i, sent_labels in enumerate(labels):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned = align_labels(sent_labels, word_ids, label2id)
        all_aligned_labels.append(aligned)

    tokenized["labels"] = all_aligned_labels
    return tokenized
