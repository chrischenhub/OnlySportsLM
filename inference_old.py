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
import shutil

access_token = "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"
cache_dir = "/root/.cache/huggingface"

#disable_caching()
RETRY_LIMIT = 5 # 设置重试次数

delete_target_dir = '/tmp/'
delete_target_prefix = 'hf_datasets'

def delete_files(file_path):
    for root, dirs, files in os.walk(file_path, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            os.remove(file_path)
        for name in dirs:
            dir_path = os.path.join(root, name)
            os.rmdir(dir_path)


def delete_prefix_folders(target_dir, target_prefix):
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


def process_data(name):

    retry_count = 0
    while retry_count < RETRY_LIMIT:
        try:
            dataset = load_dataset("Chrisneverdie/OnlySports", name,
                       split="train", token=access_token)

            print('Dataset loaded')
            break
        except Exception as e:
            retry_count += 1
  

            if retry_count >= RETRY_LIMIT:
                error_message = f"Failed to upload dataset after {RETRY_LIMIT} retries. Error: {str(e)}"
                with open("upload_error.txt", "a") as file:
                    file.write(error_message + "\n")
                print(error_message)


    retry_count = 0
    while retry_count < RETRY_LIMIT:
        try:
            dataset = dataset.map(compute_scores, batched=True, batch_size=512)
            #dataset = dataset.map(add_prefix)
            dataset = dataset.filter(lambda example: example["pred"]==1)
            dataset = dataset.select_columns(['text','url','token_count'])
            print('Dataset filtered')
            break
        except Exception as e:
            retry_count += 1
  

            if retry_count >= RETRY_LIMIT:
                error_message = f"Failed to upload dataset after {RETRY_LIMIT} retries. Error: {str(e)}"
                with open("upload_error.txt", "a") as file:
                    file.write(error_message + "\n")
                print(error_message)


    retry_count = 0
    while retry_count < RETRY_LIMIT:
        try:
            dataset.push_to_hub('Chrisneverdie/OnlySports_clean', config_name=name, data_dir=f'data/{name}', max_shard_size="4096MB",token=access_token)
            print('Dataset uploaded')
            break
        except Exception as e:
            retry_count += 1
  

            if retry_count >= RETRY_LIMIT:
                error_message = f"Failed to upload dataset after {RETRY_LIMIT} retries. Error: {str(e)}"
                with open("upload_error.txt", "a") as file:
                    file.write(error_message + "\n")
                print(error_message)

    print('done')

def loop(patten):
    for i in patten:
        process_data(i)
        delete_files(cache_dir)
        delete_prefix_folders(delete_target_dir, delete_target_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process parquet files to filter sports URLs.")
    parser.add_argument('-j', '--json', type=str, help='Path to JSON file with allow patterns list')
    parser.add_argument('-n', '--name', type=str, help='Path to JSON file with allow patterns list')



    args = parser.parse_args()
    if args.json:
        with open(args.json, 'r') as f:
            data = json.load(f)
            allow_patterns_list = data.get("patterns", default_patterns_list)
    else:
        allow_patterns_list = [args.name]

    loop(allow_patterns_list)