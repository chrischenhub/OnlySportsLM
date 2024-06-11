import torch
import pandas as pd
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np

access_token = "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"

tokenizer = AutoTokenizer.from_pretrained('Chrisneverdie/sports-text-classifier')
model = AutoModelForSequenceClassification.from_pretrained('Chrisneverdie/sports-text-classifier', torch_dtype=torch.bfloat16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
dataset = load_dataset('Chrisneverdie/sports-annotation-test',data_files={'train': 'sports_annotation_test.parquet'})

def compute_scores(batch):
    inputs = tokenizer(batch['text'], return_tensors="pt", padding="longest", truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1).float().cpu().numpy()

    #predicted_class_id = logits.argmax().item()
    batch["score"] =  logits.tolist()
   # batch['label'] = logits.argmax().tolist()
    #batch["int_score"] = [int(round(max(0, min(score, 5)))) for score in logits]
    return batch

def add_prefix(example):
    example['ypred'] = np.argmax(example['score'])
    return example

dataset = dataset.map(compute_scores, batched=True, batch_size=512)
dataset = dataset.map(add_prefix)
dataset.push_to_hub('Chrisneverdie/sports-annotation-outcome', config_name='test', private=False, max_shard_size="4096MB",token=access_token)