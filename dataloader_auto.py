import argparse
import threading
import json
import time
import psutil
import os
import concurrent.futures
from main import allow_patterns_prefix, default_patterns_list, download_dataset, upload_dataset, delete_dataset, local_download_dir
from DataGenerator import keywords
from datasets import load_dataset, disable_caching
from filelock import FileLock
import subprocess
import os

# 假设当前路径下存在文件 dataloader_uploaded.txt
uploaded_patterns_file = "dataloader_uploaded.txt"

def load_uploaded_patterns():
    if os.path.exists(uploaded_patterns_file):
        with open(uploaded_patterns_file, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def save_uploaded_pattern(pattern):
    with open(uploaded_patterns_file, 'a') as f:
        f.write(pattern + '\n')

def process_data(name):
    uploaded_patterns = load_uploaded_patterns()
    if name in uploaded_patterns:
        print(f"Pattern {name} has already been processed. Skipping.")
        return

    dataset = load_dataset_with_retry(name)
    if dataset is None:
        return  # 如果加载失败，则退出当前函数

    dataset = dataset.select_columns(['text', 'url', 'token_count'])
    print('Dataset loaded, filtering...')
    dataset = dataset.filter(lambda example: any(keyword in example["url"] for keyword in keywords), num_proc=8)
    print('Dataset filtered, uploading...')

    retry_count = 0
    while retry_count < RETRY_LIMIT:
        try:
            dataset.push_to_hub('Chrisneverdie/OnlySports', data_dir=name, private=False, max_shard_size="4096MB", token=access_token)
            print('Upload successful')
            break
        except Exception as e:
            retry_count += 1
            wait_time = 5 * 2 ** (retry_count - 1)  # 计算等待时间
            print(f"Upload failed, retrying after {wait_time} seconds... (Attempt {retry_count}/{RETRY_LIMIT})Error: {str(e)}")
            time.sleep(wait_time)  # 等待指定时间后重试

            if retry_count >= RETRY_LIMIT:
                error_message = f"Failed to upload dataset after {RETRY_LIMIT} retries. Error: {str(e)}"
                with open("upload_error.txt", "a") as file:
                    file.write(error_message + "\n")
                print(error_message)

    print('done')
    save_uploaded_pattern(name)  # 更新已处理的 pattern

if __name__ == "__main__":
    # 环境变量增加下载速度
    command = "export HF_HUB_ENABLE_HF_TRANSFER=1"
    subprocess.run(command, shell=True, check=True)

    parser = argparse.ArgumentParser(description="Process parquet files to filter sports URLs.")
    parser.add_argument("-j", "--json", type=str, help="Path to the JSON file containing patterns")
    args = parser.parse_args()

    if args.json:
        with open(args.json, 'r') as f:
            data = json.load(f)
            patterns = data.get("patterns", [])

            for pattern in patterns:
                process_data(pattern)
                clear_cache()  # 在处理每个模式后清除缓存
    else:
        print("Please provide a JSON file containing the patterns.")
