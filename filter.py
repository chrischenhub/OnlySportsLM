import argparse
import threading
import json
from main import allow_patterns_prefix, default_patterns_list, download_dataset, upload_dataset, delete_dataset, local_download_dir
from DataGenerator import keywords
import os
import concurrent.futures
from datasets import load_dataset, disable_caching
from filelock import FileLock
import pyarrow.dataset as ds
import pyarrow.parquet as pq

cache_dir = "/root/.cache/huggingface"

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

class DownloadAndFilterHandler:
    def __init__(self, patterns_list, num_proc):
        self.patterns_list = patterns_list
        self.lock = threading.Lock()
        self.downloaded_files = self.load_processed_files('download.txt')
        self.uploaded_files = self.load_processed_files('upload.txt')
        self.num_threads = num_proc

    def load_processed_files(self, file_name):
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                return set(f.read().splitlines())
        return set()

    def update_processed_files(self, file_name, file_path):
        lock_file = f"{file_name}.lock"
        with FileLock(lock_file):
            with open(file_name, 'a') as f:
                f.write(file_path + '\n')

    def process_file(self, file_path):
        try:
            if file_path in self.uploaded_files:
                print(f"File {file_path} already uploaded, skipping.")
                return

            print(f"Loading file {file_path}\n")
            dataset = load_dataset("parquet", data_files={'train': file_path})
        except Exception as e:
            log_error(f"Error loading file {file_path}: {str(e)}")
            return

        try:
            print(f"Finished loading file {file_path}, start filtering\n")
            dataset = dataset.select_columns(['text', 'url', 'dump', 'token_count'])
            dataset = dataset.filter(lambda example: any(keyword in example["url"] for keyword in keywords))
        except Exception as e:
            log_error(f"Error filtering file {file_path}: {str(e)}")
            return

        try:
            print(f"Finished filtering file {file_path}, start uploading\n")
            parts = file_path.split(os.path.sep)
            upload_dataset(dataset, str(parts[-2]) + "/" + str(parts[-1]).rstrip(".parquet"))
            self.update_processed_files('upload.txt', file_path)
        except Exception as e:
            log_error(f"Error uploading file {file_path}: {str(e)}")
            return

        try:
            print(f"file: {file_path} finished, start deleting")
            os.remove(file_path)
        except Exception as e:
            log_error(f"Error deleting file {file_path}: {str(e)}")
            return

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

        # Use ThreadPoolExecutor to process files
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_path = {executor.submit(self.process_file, path): path for path in full_paths}

            concurrent.futures.wait(future_to_path, return_when=concurrent.futures.ALL_COMPLETED)

        # Update uploaded_files set after processing
        self.uploaded_files = self.load_processed_files('upload.txt')

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
