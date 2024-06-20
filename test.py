from main import local_download_dir, allow_patterns_prefix
import os
from datasets import load_dataset

access_token = "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"


dataset = load_dataset("Chrisneverdie/OnlySports")
dataset.push_to_hub("Chrisneverdie/OnlySports", data_dir="first_dir", private=False, max_shard_size="4096MB", token=access_token)
dataset2 = load_dataset("Chrisneverdie/OnlySports")

dataset2.push_to_hub("Chrisneverdie/OnlySports", data_dir="second_dir", private=False, max_shard_size="4096MB", token=access_token)

