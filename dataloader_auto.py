import argparse
import threading
import json
from main import allow_patterns_prefix, default_patterns_list, download_dataset, upload_dataset, delete_dataset, local_download_dir
from DataGenerator import keywords
import os
import concurrent.futures
from datasets import load_dataset, disable_caching
from filelock import FileLock

access_token = "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"
RETRY_LIMIT = 5  # 设置重试次数

# 禁用缓存
disable_caching()

def process_data(name):
    dataset = load_dataset("HuggingFaceFW/fineweb", name,
                           split="train", num_proc=8)

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
            if retry_count >= RETRY_LIMIT:
                error_message = f"Failed to upload dataset after {RETRY_LIMIT} retries. Error: {str(e)}"
                with open("upload_error.txt", "a") as file:
                    file.write(error_message + "\n")
                print(error_message)
            else:
                print(f"Upload failed, retrying... (Attempt {retry_count}/{RETRY_LIMIT})")

    print('done')

def clear_cache():
    # 清除缓存的代码
    os.system('rm -rf /root/.cache/*')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process parquet files to filter sports URLs.")
    parser.add_argument("-j", "--json", type=str, help="Path to the JSON file containing patterns")
    args = parser.parse_args()

    if args.file:
        with open(args.file, 'j') as f:
            data = json.load(f)
            patterns = data.get("patterns", [])

            for pattern in patterns:
                process_data(pattern)
                clear_cache()  # 在处理每个模式后清除缓存
    else:
        print("Please provide a JSON file containing the patterns.")
