import argparse
import threading
import json
from DataGenerator import keywords
import os
import gc
import concurrent.futures
from datasets import load_dataset, disable_caching, concatenate_datasets
from filelock import FileLock
from huggingface_hub import snapshot_download
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import time
import logging


cache_dir = "/root/.cache/huggingface"
download_hub = "HuggingFaceFW/fineweb"
upload_hub = "Chrisneverdie/OnlySports"
local_download_dir = "./downloads/test/"
access_token = "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"
max_disk_usage = 100 * 1024 * 1024 * 1024 * 10  # 1000GB
allow_patterns_prefix = "data/"
upload_folder = "test"
default_patterns_list = "CC-MAIN-2023-40"
upload_log = "uploaded.txt"

def update_processed_files(file_name, file_path):
    lock_file = f"{file_name}.lock"
    with FileLock(lock_file):
        with open(file_name, 'a') as f:
            f.write(file_path + '\n')

def load_processed_files(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            return set(f.read().splitlines())
    return set()

def filter_dataset(dataset, keywords):
    return dataset.filter(lambda example: any(keyword in example["url"] for keyword in keywords))

def process_and_filter_files(full_paths, pattern, dir_name):
    all_filtered_datasets = []
    for full_path in full_paths:
        if full_path.endswith(".parquet"):
            try:
                dataset = load_dataset('parquet', data_files=full_path, split='train', num_proc=8)
                dataset = dataset.select_columns(['text', 'url', 'token_count'])
                filtered_dataset = filter_dataset(dataset, keywords)
                all_filtered_datasets.append(filtered_dataset)
                os.remove(full_path)
                gc.collect()
            except Exception as e:
                log_error(f"Error processing file {full_path}: {str(e)}")
                continue

    data_dir = pattern + "/" + dir_name + "_6.25_ver"
    try:
        concatenated_dataset = concatenate_datasets(all_filtered_datasets)
        concatenated_dataset.push_to_hub(upload_hub, data_dir=data_dir, private=False, max_shard_size="4096MB", token=access_token)
    except Exception as e:
        log_error(f"Error uploading dataset from {full_paths}: {str(e)}")

    for full_path in full_paths:
        update_processed_files(full_path, upload_log)

def download_dataset(allow_patterns):
    try:
        filepath = snapshot_download(
            download_hub,
            repo_type="dataset",
            local_dir=local_download_dir,
            allow_patterns=allow_patterns_prefix + allow_patterns + "/*")
        return filepath
    except Exception as e:
        log_error(f"Error downloading dataset for pattern {allow_patterns}: {str(e)}")
        return None


def upload_dataset(dataset, data_dir):
    max_retries = 5
    retry_delay = 5  # 初始重试延迟时间

    for retry in range(max_retries):
        try:
            dataset.push_to_hub(upload_hub, data_dir=data_dir, private=False, max_shard_size="4096MB", token=access_token)
            print(f"Dataset successfully uploaded to {upload_hub} with data_dir {data_dir}")
            return
        except Exception as e:
            if retry < max_retries - 1:  # 如果不是最后一次尝试
                print(f"Error uploading dataset to {upload_hub} with data_dir {data_dir}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 增加重试延迟时间
            else:
                log_error(f"Failed to upload dataset to {upload_hub} with data_dir {data_dir} after {max_retries} attempts: {str(e)}")
                return

def delete_files(file_path):
    try:
        for root, dirs, files in os.walk(file_path, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                os.remove(file_path)
            for name in dirs:
                dir_path = os.path.join(root, name)
                os.rmdir(dir_path)
    except Exception as e:
        log_error(f"Error deleting files in {file_path}: {str(e)}")

def log_error(error_message):
    error_file = 'error.txt'
    with FileLock(error_file + '.lock'):
        if not os.path.exists(error_file):
            with open(error_file, 'w') as f:
                f.write("Error Log\n")
        with open(error_file, 'a') as f:
            f.write(error_message + '\n')

def split_list_into_chunks(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

class DownloadAndFilterHandler:
    def __init__(self, patterns_list, num_proc, chunk_size):
        self.patterns_list = patterns_list
        self.lock = threading.Lock()
        self.downloaded_files = load_processed_files('download.txt')
        self.uploaded_files = load_processed_files('upload.txt')
        self.num_threads = num_proc
        self.chunk_size = chunk_size

    import logging

    def download_filter(self, pattern):
        pattern_path = local_download_dir + allow_patterns_prefix + pattern + "/"
        print(f"Starting download and filter process for pattern: {pattern}")

        if pattern in self.downloaded_files:
            print(f"Pattern {pattern} already downloaded, skipping.")
        else:
            try:
                print(f"Attempting to download dataset for pattern {pattern}")
                filepath = download_dataset(pattern)
                if filepath:
                    print(f"Dataset downloaded successfully for pattern {pattern}")
                    update_processed_files('download.txt', pattern)
                else:
                    print(f"Failed to download dataset for pattern {pattern}")
            except Exception as e:
                error_message = f"Error downloading dataset for pattern {pattern}: {str(e)}"
                log_error(error_message)

        file_names = [f for f in os.listdir(pattern_path)]
        full_paths = [pattern_path + filename for filename in file_names]

        chunks = split_list_into_chunks(full_paths, self.chunk_size)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            try:
                print(f"Processing chunks for pattern {pattern}")
                future_to_chunk = {executor.submit(process_and_filter_files, chunk, pattern, chunk[0] + "_to_" + chunk[-1]): chunk for chunk in chunks}



                # Wait for all tasks to complete
                concurrent.futures.wait(future_to_chunk.keys())

                # Optionally, you can print the status of each task
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        future.result()  # This will raise an exception if the task failed
                    except Exception as e:
                        error_message = f"Error processing chunk {chunk} for pattern {pattern}: {str(e)}"
                        print(error_message)
                        log_error(error_message)
            except Exception as e:
                error_message = f"Error processing chunks for pattern {pattern}: {str(e)}"
                print(error_message)
                log_error(error_message)

        self.uploaded_files = load_processed_files('upload.txt')
        remaining_paths = [path for path in full_paths if path not in self.uploaded_files]


        if remaining_paths:
            logging.info("Reprocessing remaining files: " + str(remaining_paths))
            self.download_filter(pattern)
        else:
            logging.info(f"No remaining files to process for pattern {pattern}")

        logging.info("Download and filter process for pattern completed.")

    def run(self):
        for pattern in self.patterns_list:
            self.download_filter(pattern)
            print("All Finished, Start Deleting")
            delete_files(cache_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset handling script.")
    parser.add_argument('-t', '--threads', type=int, default=3, help='Number of threads in the thread pool')
    parser.add_argument('-j', '--json', type=str, help='Path to JSON file with allow patterns list')
    parser.add_argument('-c', '--chunk', type=int, help='Chunk size', default=5)

    return parser.parse_args()

def main():
    args = parse_args()

    if args.json:
        with open(args.json, 'r') as f:
            data = json.load(f)
            allow_patterns_list = data.get("patterns", default_patterns_list)
    else:
        allow_patterns_list = default_patterns_list

    handler = DownloadAndFilterHandler(allow_patterns_list, args.threads, args.chunk)
    handler.run()

if __name__ == "__main__":
    main()
