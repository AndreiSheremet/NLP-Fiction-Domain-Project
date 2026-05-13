from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, AutoConfig, set_seed)
import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

# Set random seeds
set_seed(42)

# Define hyperparameters (e.g., learning_rate, num_train_epochs, model_name)
learning_rate = 2e-5
num_train_epochs = 4
model_name = model_name = "bert-base-cased"
import gc



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

train_dataset = Dataset.from_dict({
    "tokens": train_sentences,
    "ner_tags": train_labels
})
dev_dataset = Dataset.from_dict({
    "tokens": dev_sentences,
    "ner_tags": dev_labels
})

test_dataset = Dataset.from_dict({
    "tokens": test_sentences,
    "ner_tags": test_labels
})

# Load the tokenizer and model config
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
config = AutoConfig.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# Initialize the model with AutoModelForTokenClassification
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    config=config
)

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

def tokenize_and_align_labels(sentences, labels, tokenizer, label2id):
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

def preprocess(input):
    return tokenize_and_align_labels(
        input["tokens"],
        input["ner_tags"],
        tokenizer,
        label2id
    )

tokenized_train_dataset = train_dataset.map(preprocess, batched=True)
tokenized_train_dataset = tokenized_train_dataset.remove_columns(["tokens", "ner_tags"])
tokenized_train_dataset.set_format(type="torch")

tokenized_dev_dataset = dev_dataset.map(preprocess, batched=True)
tokenized_dev_dataset = tokenized_dev_dataset.remove_columns(["tokens", "ner_tags"])
tokenized_dev_dataset.set_format(type="torch")

tokenized_test_dataset = test_dataset.map(preprocess, batched=True)
tokenized_test_dataset = tokenized_test_dataset.remove_columns(["tokens", "ner_tags"])
tokenized_test_dataset.set_format(type="torch")

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

train_dataloader = DataLoader(
    tokenized_train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=data_collator
)

dev_dataloader = DataLoader(
    tokenized_dev_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=data_collator
)
# Move model to device (CPU/GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(device)
# Create optimizer (e.g. AdamW)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# import torch_directml
# device = torch_directml.device() 
# model.to(device)

# optimizer = torch.optim.SGD(
#     model.parameters(), 
#     lr=5e-5, 
#     momentum=0.9, 
#     nesterov=True, 
#     weight_decay=0.01
# )

# Implement the training loop
import torch
model.train()

for epoch in range(num_train_epochs):
    total_loss = 0
    
    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}")
    
    for step, batch in pbar:
        #move batch to device, move every tensor(datapoints) in it to GPU/CPU
        batch = {k: v.to(device) for k, v in batch.items()}

        #forward pass
        #tuple-like object for hugging face 
        #loss calcukated automatically cuz we included 'labels' in the batch. 
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
         #backward pass
        loss.backward()  
        optimizer.step()
        #track loss
        total_loss += loss.item()
        # update progress bar with loss
        pbar.set_postfix({"loss": loss.item()})

    torch.cuda.empty_cache() # clear the backend just in case
    gc.collect()
    #print average loss
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} average training loss: {avg_train_loss:.4f}")


def get_predictions(words, model, tokenizer, id2label, device):
    if not words:
        return []
    
    inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().tolist()
    word_ids = inputs.word_ids(batch_index=0)
    
    pred_labels = []
    last_word_idx = None
    for i, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != last_word_idx:
            pred_labels.append(id2label[predictions[i]])
            last_word_idx = word_idx

    return pred_labels

def write_sentence(f_out, lines, labels):
    for i, line in enumerate(lines):
        parts = line.split('\t')
        if len(parts) >= 3:
            parts[2] = labels[i]
            f_out.write("\t".join(parts) + "\n")
        else:
            f_out.write(line + "\n")

def predict_and_save(input_path, output_path, model, tokenizer, id2label, device):
    model.eval()
    
    with open(input_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out, \
         tqdm(total=total_lines, desc="Predicting") as pbar:
        
        current_words = []
        current_lines = []

        for line in f_in:
            pbar.update(1)
            raw_line = line.rstrip("\n\r")
            if not raw_line or raw_line.startswith("#"):
                if current_words:
                    labels = get_predictions(current_words, model, tokenizer, id2label, device)
                    write_sentence(f_out, current_lines, labels)
                    current_words, current_lines = [], []
                f_out.write(line)
                continue

            parts = raw_line.split('\t')
            if len(parts) >= 2:
                current_words.append(parts[1])
                current_lines.append(raw_line)

        if current_words:
            labels = get_predictions(current_words, model, tokenizer, id2label, device)
            write_sentence(f_out, current_lines, labels)

predict_and_save(
    'project/en_ewt-ud-test.iob2',
    'project/en_ewt-ud-test-predictions.iob2',
    model, tokenizer, id2label, device
)
    