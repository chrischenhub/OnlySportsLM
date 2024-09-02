import torch
import pandas as pd
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
from datasets import disable_caching
import os
from main import allow_patterns_prefix, default_patterns_list
import json

disable_caching()

access_token = "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"
cache_dir = "/root/.cache/huggingface"




tokenizer = AutoTokenizer.from_pretrained('Chrisneverdie/sports-text-classifier')
model = AutoModelForSequenceClassification.from_pretrained('Chrisneverdie/sports-text-classifier', torch_dtype=torch.bfloat16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# def compute_scores(batch):
#     inputs = tokenizer(batch['text'], return_tensors="pt", padding="longest", truncation=True).to(device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits.squeeze(-1).float().cpu().numpy()

#     batch["score"] =  logits.tolist()
#     return batch

# def add_prefix(example):
#     example['pred'] = np.argmax(example['score'])
#     return example



def compute_scores(batch):
    inputs = tokenizer(batch['text'], return_tensors="pt", padding="longest", truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs).logits.argmax(dim=-1).cpu().numpy()

    batch["pred"] =  outputs
    return batch

dataset = load_dataset('Chrisneverdie/sports-annotation',data_files={'train': 'train.parquet'})
dataset = dataset.map(compute_scores, batched=True, batch_size=512)
#dataset = dataset.map(add_prefix)
dataset = dataset.filter(lambda example: example["pred"]==1)
#dataset = dataset.select_columns(['text','url','token_count'])
print('Dataset filtered')

dataset.push_to_hub('Chrisneverdie/OnlySports_clean', config_name='test', data_dir=f'data/test', private=False, max_shard_size="4096MB",token=access_token)
