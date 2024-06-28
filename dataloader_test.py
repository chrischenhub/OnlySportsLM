import argparse
import threading
import json
from main import allow_patterns_prefix, default_patterns_list, download_dataset, upload_dataset, delete_dataset, local_download_dir
from DataGenerator import keywords
import os
import concurrent.futures
from datasets import load_dataset, disable_caching
from filelock import FileLock
from datasets import disable_caching
disable_caching()

access_token = "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"

#disable_caching()
RETRY_LIMIT = 8  # 设置重试次数

def process_data(name):
    dataset = load_dataset("HuggingFaceFW/fineweb", name,
                        split="train", num_proc=8)

    dataset = dataset.select_columns(['text', 'url', 'token_count'])
    print('Dataset loaded, filtering...')
    dataset = dataset.filter(lambda example: any(keyword in example["url"] for keyword in keywords),num_proc=8)
    print('Dataset filtered, uploading...')
    dataset.push_to_hub('Chrisneverdie/OnlySports', data_dir=name, private=False, max_shard_size="4096MB", token=access_token)

    print('done')
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

    print('done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process parquet files to filter sports URLs.")
    parser.add_argument("-n", "--name", type=str, help="Target pattern in the hub")
    args = parser.parse_args()
    process_data(args.name)