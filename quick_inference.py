import torch
import pandas as pd
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
import os
import shutil

access_token = "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"

tokenizer = AutoTokenizer.from_pretrained('Chrisneverdie/sports-text-classifier')
model = AutoModelForSequenceClassification.from_pretrained('Chrisneverdie/sports-text-classifier', torch_dtype=torch.bfloat16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
dataset = load_dataset('Chrisneverdie/sports-annotation-test',data_files={'train': 'nonsport.parquet'})
delete_target_dir = '/tmp/'
delete_target_prefix = 'hf_datasets'

def delete_hf_datasets_folders(target_dir, target_prefix):
    # 遍历目标目录下的所有文件和文件夹
    for root, dirs, files in os.walk(target_dir, topdown=False):
        for dir_name in dirs:
            if dir_name.startswith(target_prefix):
                # 构建完整路径
                full_path = os.path.join(root, dir_name)
                try:
                    # 删除文件夹及其内容
                    shutil.rmtree(full_path)
                    print(f"Deleted directory: {full_path}")
                except Exception as e:
                    print(f"Failed to delete {full_path}: {e}")

def compute_scores(batch):
    inputs = tokenizer(batch['text'], return_tensors="pt", padding="longest", truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1).float().cpu().numpy()

    batch["score"] =  logits.tolist()
    return batch

def add_prefix(example):
    example['ypred'] = np.argmax(example['score'])
    return example

dataset = dataset.map(compute_scores, batched=True, batch_size=512)
dataset = dataset.map(add_prefix)
dataset = dataset.filter(lambda example: example["ypred"]==1)
dataset.push_to_hub('Chrisneverdie/sports-annotation-outcome', config_name='nonsport_outcome', private=False, max_shard_size="4096MB",token=access_token)
delete_hf_datasets_folders(delete_target_dir, delete_target_prefix)