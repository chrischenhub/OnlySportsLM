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

cache_dir = "/root/.cache/huggingface"
download_hub = "HuggingFaceFW/fineweb"
upload_hub = "Chrisneverdie/OnlySports"
local_download_dir = "./downloads/test/"
access_token = "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"
max_disk_usage = 100 * 1024 * 1024 * 1024 * 10 # 1000GB
allow_patterns_prefix = "data/"
upload_folder = "test"
default_patterns_list = ['CC-MAIN-2013-20/000_00000.parquet', 'CC-MAIN-2013-20/000_00001.parquet', 'CC-MAIN-2013-20/000_00002.parquet'
                       'CC-MAIN-2013-48', 'CC-MAIN-2014-10', 'CC-MAIN-2014-15', 'CC-MAIN-2014-23', 'CC-MAIN-2014-35', 'CC-MAIN-2014-41', 'CC-MAIN-2014-42', 'CC-MAIN-2014-49', 'CC-MAIN-2014-52', 'CC-MAIN-2015-06', 'CC-MAIN-2015-11', 'CC-MAIN-2015-14', 'CC-MAIN-2015-18', 'CC-MAIN-2015-22', 'CC-MAIN-2015-27', 'CC-MAIN-2015-32', 'CC-MAIN-2015-35', 'CC-MAIN-2015-40', 'CC-MAIN-2015-48', 'CC-MAIN-2016-07', 'CC-MAIN-2016-18', 'CC-MAIN-2016-22', 'CC-MAIN-2016-26', 'CC-MAIN-2016-30', 'CC-MAIN-2016-36', 'CC-MAIN-2016-40', 'CC-MAIN-2016-44', 'CC-MAIN-2016-50', 'CC-MAIN-2017-04', 'CC-MAIN-2017-09', 'CC-MAIN-2017-13', 'CC-MAIN-2017-17', 'CC-MAIN-2017-22', 'CC-MAIN-2017-26', 'CC-MAIN-2017-30', 'CC-MAIN-2017-34', 'CC-MAIN-2017-39', 'CC-MAIN-2017-43', 'CC-MAIN-2017-47', 'CC-MAIN-2017-51', 'CC-MAIN-2018-05', 'CC-MAIN-2018-09', 'CC-MAIN-2018-13', 'CC-MAIN-2018-17', 'CC-MAIN-2018-22', 'CC-MAIN-2018-26', 'CC-MAIN-2018-30', 'CC-MAIN-2018-34', 'CC-MAIN-2018-39', 'CC-MAIN-2018-43', 'CC-MAIN-2018-47', 'CC-MAIN-2018-51', 'CC-MAIN-2019-04', 'CC-MAIN-2019-09', 'CC-MAIN-2019-13', 'CC-MAIN-2019-18', 'CC-MAIN-2019-22', 'CC-MAIN-2019-26', 'CC-MAIN-2019-30', 'CC-MAIN-2019-35', 'CC-MAIN-2019-39', 'CC-MAIN-2019-43', 'CC-MAIN-2019-47', 'CC-MAIN-2019-51', 'CC-MAIN-2020-05', 'CC-MAIN-2020-10', 'CC-MAIN-2020-16', 'CC-MAIN-2020-24', 'CC-MAIN-2020-29', 'CC-MAIN-2020-34', 'CC-MAIN-2020-40', 'CC-MAIN-2020-45', 'CC-MAIN-2020-50', 'CC-MAIN-2021-04', 'CC-MAIN-2021-10', 'CC-MAIN-2021-17', 'CC-MAIN-2021-21', 'CC-MAIN-2021-25', 'CC-MAIN-2021-31', 'CC-MAIN-2021-39', 'CC-MAIN-2021-43', 'CC-MAIN-2021-49', 'CC-MAIN-2022-05', 'CC-MAIN-2022-21', 'CC-MAIN-2022-27', 'CC-MAIN-2022-33', 'CC-MAIN-2022-40', 'CC-MAIN-2022-49', 'CC-MAIN-2023-06', 'CC-MAIN-2023-14', 'CC-MAIN-2023-23', 'CC-MAIN-2023-40', 'CC-MAIN-2023-50', 'CC-MAIN-2024-10', 'CC-MAIN-2024-18']
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

def process_and_filter_files(full_paths, pattern, index):
    all_filtered_datasets = []
    for full_path in full_paths:
        if full_path.endswith(".parquet"):
            dataset = load_dataset('parquet', data_files=full_path, split='train',num_proc=8)
            dataset = dataset.select_columns(['text','url','token_count'])

            filtered_dataset = filter_dataset(dataset, keywords)
            all_filtered_datasets.append(filtered_dataset)

            os.remove(full_path)
            gc.collect()

    data_dir = pattern + "/" + index + "6.25 ver"
    concatenated_dataset = concatenate_datasets(all_filtered_datasets)
    concatenated_dataset.push_to_hub(upload_hub, data_dir=data_dir, private=False, max_shard_size="4096MB", token=access_token)

    for full_path in full_paths:
        update_processed_files(full_path, upload_log)


def download_dataset(allow_patterns):
    filepath = snapshot_download(
        download_hub,
        repo_type="dataset",
        local_dir=local_download_dir,
        allow_patterns=allow_patterns_prefix + allow_patterns + "/*")
    return filepath

def upload_dataset(dataset, data_dir):
    dataset.push_to_hub(upload_hub, data_dir=data_dir, private=False, max_shard_size="4096MB", token=access_token)

def delete_files(file_path):
    for root, dirs, files in os.walk(file_path, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            os.remove(file_path)
        for name in dirs:
            dir_path = os.path.join(root, name)
            os.rmdir(dir_path)

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




    def download_filter(self, pattern):
        pattern_path = local_download_dir + allow_patterns_prefix + pattern + "/"

        if pattern in self.downloaded_files:
            print(f"Pattern {pattern} already downloaded, skipping.")
        else:
            try:
                download_dataset(pattern)
                self.update_processed_files('download.txt', pattern)
            except Exception as e:
                log_error(f"Error downloading dataset for pattern {pattern}: {str(e)}")
                return

        file_names = [f for f in os.listdir(pattern_path)]
        full_paths = [pattern_path + filename for filename in file_names]

        print("full_paths: " + str(full_paths))

        # 将full_paths分成每chunk_size个一组
        chunks = split_list_into_chunks(full_paths, self.chunk_size)

        # 使用多线程处理每个子列表
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(process_and_filter_files, chunks, [pattern] * len(chunks))


        # Update uploaded_files set after processing
        self.uploaded_files = load_processed_files('upload.txt')

        # Remove already uploaded files from full_paths
        remaining_paths = [path for path in full_paths if path not in self.uploaded_files]

        # If there are remaining files, call download_filter again
        if remaining_paths:
            print("Reprocessing remaining files: " + str(remaining_paths))
            self.download_filter(pattern)

    def run(self):
        for pattern in self.patterns_list:
            self.download_filter(pattern)
            print("All Finished, Start Deleting")
            delete_files(cache_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset handling script.")
    parser.add_argument('-t', '--threads', type=int, default=3, help='Number of threads in the thread pool')
    parser.add_argument('-j', '--json', type=str, help='Path to JSON file with allow patterns list')
    parser.add_argument('-c', '--chunk', type=int, help='Chunk size', default= 5)

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