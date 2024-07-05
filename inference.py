import threading
import time
import os
import shutil
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

access_token = "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"

cache_dir = "/root/.cache/huggingface"
log_file_path = "processed.log"

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

def compute_scores(batch):
    inputs = tokenizer(batch['text'], return_tensors="pt", padding="longest", truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs).logits.argmax(dim=-1).cpu().numpy()
    batch["pred"] = outputs
    return batch

def process_data(name, stop_event):
    retry_count = 0
    while retry_count < RETRY_LIMIT:
        if stop_event.is_set():
            return
        try:
            dataset = load_dataset("Chrisneverdie/OnlySports", name, split="train", num_proc=8, token=access_token)
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
        if stop_event.is_set():
            return
        try:
            dataset = dataset.map(compute_scores, batched=True, batch_size=512)
            dataset = dataset.filter(lambda example: example["pred"] == 1)
            dataset = dataset.select_columns(['text', 'url', 'token_count'])
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
        if stop_event.is_set():
            return
        try:
            dataset.push_to_hub('Chrisneverdie/OnlySports_clean', config_name=name, data_dir=f'data/{name}', max_shard_size="4096MB", token=access_token)
            print('Dataset uploaded')
            break
        except Exception as e:
            retry_count += 1
            if retry_count >= RETRY_LIMIT:
                error_message = f"Failed to upload dataset after {RETRY_LIMIT} retries. Error: {str(e)}"
                with open("upload_error.txt", "a") as file:
                    file.write(error_message + "\n")
                print(error_message)

    # 记录处理完成的name到log文件中
    with open(log_file_path, "a") as log_file:
        log_file.write(name + "\n")

    print('done')

def loop(pattern, stop_event):
    for i in pattern:
        if stop_event.is_set():
            return
        process_data(i, stop_event)
        delete_files(cache_dir)
        delete_prefix_folders(delete_target_dir, delete_target_prefix)

def get_dir_size(dir_path):
    total_size = shutil.disk_usage(dir_path).used
    return total_size

def monitor_cache_dir(stop_event, size_change_event, monitor_interval=180):
    initial_size = get_dir_size(cache_dir)
    while not stop_event.is_set():
        time.sleep(5)
        current_size = get_dir_size(cache_dir)
        if current_size != initial_size:
            initial_size = current_size
            size_change_event.set()
        else:
            size_change_event.clear()
        time.sleep(monitor_interval)

def read_processed_log():
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as log_file:
            processed = log_file.read().splitlines()
    else:
        processed = []
    return processed

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

    processed_names = read_processed_log()
    allow_patterns_list = [name for name in allow_patterns_list if name not in processed_names]

    stop_event = threading.Event()
    size_change_event = threading.Event()

    process_thread = threading.Thread(target=loop, args=(allow_patterns_list, stop_event))
    monitor_thread = threading.Thread(target=monitor_cache_dir, args=(stop_event, size_change_event))

    process_thread.start()
    monitor_thread.start()

    while process_thread.is_alive():
        if size_change_event.is_set():
            continue
        else:
            print("Disk usage unchanged, restarting process thread...")
            stop_event.set()
            process_thread.join()
            stop_event.clear()
            process_thread = threading.Thread(target=loop, args=(allow_patterns_list, stop_event))
            process_thread.start()

    stop_event.set()
    monitor_thread.join()
