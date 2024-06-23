import argparse
import threading
import json
from main import allow_patterns_prefix, default_patterns_list, download_dataset, upload_dataset, delete_dataset, local_download_dir
from DataGenerator import keywords
import os
import concurrent.futures
from datasets import load_dataset, disable_caching
from filelock import FileLock

#disable_caching()

cache_dir = "/root/.cache/huggingface"

dataset = load_dataset("HuggingFaceFW/fineweb", "CC-MAIN-2024-18",
                    split="train", num_proc=4)


dataset = dataset.select_columns(['text', 'url', 'dump', 'token_count'])
dataset = dataset.filter(lambda example: any(keyword in example["url"] for keyword in keywords),num_proc=4)
upload_dataset(dataset, "CC-MAIN-2019-04")