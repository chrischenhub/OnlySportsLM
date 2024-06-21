import argparse
import threading
import json
from main import allow_patterns_prefix, default_patterns_list, download_dataset, upload_dataset, delete_dataset, local_download_dir
from DataGenerator import keywords
import os
import concurrent.futures
from datasets import load_dataset
from filelock import FileLock

cache_dir = "/root/.cache/huggingface"

def delete_files(file_path):
    for root, dirs, files in os.walk(file_path, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            os.remove(file_path)
        for name in dirs:
            dir_path = os.path.join(root, name)
            os.rmdir(dir_path)



class DownloadAndFilterHandler:
    def __init__(self, patterns_list):
        self.patterns_list = patterns_list
        self.lock = threading.Lock()
        self.downloaded_files = self.load_processed_files('download.txt')
        self.uploaded_files = self.load_processed_files('upload.txt')

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
        if file_path in self.uploaded_files:
            print(f"File {file_path} already uploaded, skipping.")
            return

        print(f"Loading file {file_path}\n")
        dataset = load_dataset("parquet", data_files={'train': file_path})
        print(f"Finished loading file {file_path}, start filtering\n")
        dataset = dataset.select_columns(['text', 'url', 'dump', 'token_count'])
        dataset = dataset.filter(lambda example: any(keyword in example["url"] for keyword in keywords))
        print(f"Finished filtering file {file_path}, start uploading\n")
        parts = file_path.split(os.path.sep)
        upload_dataset(dataset, str(parts[-2]) + "_" + str(parts[-1]))
        self.update_processed_files('upload.txt', file_path)
        dataset.cleanup_cache_files()

    def download_filter(self, pattern):
        pattern_path = local_download_dir + allow_patterns_prefix + pattern + "/"

        if pattern in self.downloaded_files:
            print(f"Pattern {pattern} already downloaded, skipping.")
        else:
            download_dataset(pattern)
            self.update_processed_files('download.txt', pattern)

        file_names = [f for f in os.listdir(pattern_path)]
        full_paths = [pattern_path + filename for filename in file_names]

        # 使用线程池处理文件
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_path = {executor.submit(self.process_file, path): path for path in full_paths}

            concurrent.futures.wait(future_to_path, return_when=concurrent.futures.ALL_COMPLETED)

        delete_files(pattern_path)


    def run(self):
        for pattern in self.patterns_list:
            self.download_filter(pattern)

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset handling script.")
    parser.add_argument('-j', '--json', type=str, help='Path to JSON file with allow patterns list')
    return parser.parse_args()

def main():
    args = parse_args()

    if args.json:
        with open(args.json, 'r') as f:
            data = json.load(f)
            allow_patterns_list = data.get("patterns", default_patterns_list)
    else:
        allow_patterns_list = default_patterns_list

    handler = DownloadAndFilterHandler(allow_patterns_list)
    handler.run()

if __name__ == "__main__":
    main()
