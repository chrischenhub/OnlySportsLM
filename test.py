from main import local_download_dir, allow_patterns_prefix
import os
from datasets import load_dataset
import shutil
access_token = "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"


def delete_dataset(filepath):
    if os.path.exists(filepath):
        if os.path.isfile(filepath):
            os.remove(filepath)
        else:
            shutil.rmtree(filepath)

delete_dataset("downloads/test/data/pattern/test2")