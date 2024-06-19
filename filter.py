import argparse
import test_classifier as cl
import threading

import concurrent.futures
import json
from main import access_token,default_patterns_list, download_dataset, upload_dataset
from DataGenerator import keywords

class DownloadAndFilterHandler:
    def __init__(self, patterns_list, num_threads):
        self.patterns_list = patterns_list
        self.num_threads = num_threads
        self.lock = threading.Lock()

    def download_filter(self, pattern):
        filepath = download_dataset(pattern)
        dataset = cl.my_load_dataset(filepath)
        dataset = dataset.select_columns(['text','url','dump','token_count'])
        dataset = dataset.filter(lambda example: any(keyword in example["url"] for keyword in keywords))
        upload_dataset(dataset)

    def run(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for pattern in self.patterns_list:
                futures.append(executor.submit(self.download_filter, pattern))

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Exception occurred: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset handling script.")
    parser.add_argument('-t', '--threads', type=int, default=3, help='Number of threads in the thread pool')
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

    print("pattern list: \n" + str(allow_patterns_list))

    handler = DownloadAndFilterHandler(allow_patterns_list, args.threads)
    handler.run()


if __name__ == "__main__":
    main()
