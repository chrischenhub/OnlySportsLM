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

#disable_caching()
HF_HUB_ENABLE_HF_TRANSFER=1

cache_dir = "/root/.cache/huggingface"

dataset = load_dataset("HuggingFaceFW/fineweb", "CC-MAIN-2024-10",
            streaming=True, split="train")


dataset = dataset.select_columns(['text', 'url', 'dump', 'token_count'])
dataset = dataset.filter(lambda example: any(keyword in example["url"] for keyword in keywords),num_proc=8)
dataset.push_to_hub('Chrisneverdie/OnlySports', data_dir='CC-MAIN-2024-10', private=False, max_shard_size="4096MB", token=access_token)
