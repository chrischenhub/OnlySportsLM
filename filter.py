import argparse
import test_classifier as cl
import threading
import json
from main import allow_patterns_prefix, default_patterns_list, download_dataset, upload_dataset, delete_dataset, local_download_dir
from DataGenerator import keywords
import os
import concurrent.futures

class DownloadAndFilterHandler  :
    def __init__(self, patterns_list):
        self.patterns_list = patterns_list
        self.lock = threading.Lock()

    def process_file(self, file_path):
        dataset = cl.my_load_dataset(file_path)
        dataset = dataset.select_columns(['text', 'url', 'dump', 'token_count'])
        dataset = dataset.filter(lambda example: any(keyword in example["url"] for keyword in keywords))
        upload_dataset(dataset)
        delete_dataset(file_path)

    def download_filter(self, pattern):
        download_dataset(pattern)

        #change file path / change os.path.isfile(f)
        #add log
        pattern_path = local_download_dir + allow_patterns_prefix + pattern + "/"
        file_names = [f for f in os.listdir(pattern_path)]
        full_paths = [pattern_path + filename for filename in file_names]

        # 使用线程池处理文件
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self.process_file, full_paths)

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
