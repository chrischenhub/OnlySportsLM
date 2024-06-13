import test_classifier as cl
from huggingface_hub import snapshot_download
import threading
import os
import shutil
import json

download_hub = "HuggingFaceFW/fineweb"
upload_hub = "Chrisneverdie/sports-annotation-outcome"
local_dir = "./downloads/test/"
access_token = "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"


allow_patterns_list = ['CC-MAIN-2013-20/000_00000.parquet', 'CC-MAIN-2013-20/000_00001.parquet', 'CC-MAIN-2013-20/000_00002.parquet'
                       'CC-MAIN-2013-48', 'CC-MAIN-2014-10', 'CC-MAIN-2014-15', 'CC-MAIN-2014-23', 'CC-MAIN-2014-35', 'CC-MAIN-2014-41', 'CC-MAIN-2014-42', 'CC-MAIN-2014-49', 'CC-MAIN-2014-52', 'CC-MAIN-2015-06', 'CC-MAIN-2015-11', 'CC-MAIN-2015-14', 'CC-MAIN-2015-18', 'CC-MAIN-2015-22', 'CC-MAIN-2015-27', 'CC-MAIN-2015-32', 'CC-MAIN-2015-35', 'CC-MAIN-2015-40', 'CC-MAIN-2015-48', 'CC-MAIN-2016-07', 'CC-MAIN-2016-18', 'CC-MAIN-2016-22', 'CC-MAIN-2016-26', 'CC-MAIN-2016-30', 'CC-MAIN-2016-36', 'CC-MAIN-2016-40', 'CC-MAIN-2016-44', 'CC-MAIN-2016-50', 'CC-MAIN-2017-04', 'CC-MAIN-2017-09', 'CC-MAIN-2017-13', 'CC-MAIN-2017-17', 'CC-MAIN-2017-22', 'CC-MAIN-2017-26', 'CC-MAIN-2017-30', 'CC-MAIN-2017-34', 'CC-MAIN-2017-39', 'CC-MAIN-2017-43', 'CC-MAIN-2017-47', 'CC-MAIN-2017-51', 'CC-MAIN-2018-05', 'CC-MAIN-2018-09', 'CC-MAIN-2018-13', 'CC-MAIN-2018-17', 'CC-MAIN-2018-22', 'CC-MAIN-2018-26', 'CC-MAIN-2018-30', 'CC-MAIN-2018-34', 'CC-MAIN-2018-39', 'CC-MAIN-2018-43', 'CC-MAIN-2018-47', 'CC-MAIN-2018-51', 'CC-MAIN-2019-04', 'CC-MAIN-2019-09', 'CC-MAIN-2019-13', 'CC-MAIN-2019-18', 'CC-MAIN-2019-22', 'CC-MAIN-2019-26', 'CC-MAIN-2019-30', 'CC-MAIN-2019-35', 'CC-MAIN-2019-39', 'CC-MAIN-2019-43', 'CC-MAIN-2019-47', 'CC-MAIN-2019-51', 'CC-MAIN-2020-05', 'CC-MAIN-2020-10', 'CC-MAIN-2020-16', 'CC-MAIN-2020-24', 'CC-MAIN-2020-29', 'CC-MAIN-2020-34', 'CC-MAIN-2020-40', 'CC-MAIN-2020-45', 'CC-MAIN-2020-50', 'CC-MAIN-2021-04', 'CC-MAIN-2021-10', 'CC-MAIN-2021-17', 'CC-MAIN-2021-21', 'CC-MAIN-2021-25', 'CC-MAIN-2021-31', 'CC-MAIN-2021-39', 'CC-MAIN-2021-43', 'CC-MAIN-2021-49', 'CC-MAIN-2022-05', 'CC-MAIN-2022-21', 'CC-MAIN-2022-27', 'CC-MAIN-2022-33', 'CC-MAIN-2022-40', 'CC-MAIN-2022-49', 'CC-MAIN-2023-06', 'CC-MAIN-2023-14', 'CC-MAIN-2023-23', 'CC-MAIN-2023-40', 'CC-MAIN-2023-50', 'CC-MAIN-2024-10', 'CC-MAIN-2024-18']

download_lock = threading.Lock()  # Global lock for download, for disk capacity concern


def download_dataset(allow_patterns):
    with download_lock:
        filepath = snapshot_download(
            download_hub,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns="data/" + allow_patterns)
    return filepath


def upload_dataset(dataset):
    dataset.push_to_hub(upload_hub, config_name='test', private=False, max_shard_size="4096MB", token=access_token)


def delete_dataset(filepath):
    if os.path.exists(filepath):
        if os.path.isfile(filepath):
            os.remove(filepath)
        else:
            shutil.rmtree(filepath)


class DatasetHandler:
    def __init__(self, patterns_list):
        self.patterns_list = patterns_list
        self.current_index = 0
        self.lock = threading.Lock()

    def process_and_download(self):
        while self.current_index < len(self.patterns_list):
            with self.lock:
                current_pattern = self.patterns_list[self.current_index]
                print("\ncurrent pattern is :" + current_pattern + "\n")
                self.current_index += 1

            filepath = download_dataset(current_pattern)
            dataset = cl.my_load_dataset(filepath)

            # Start the next download in a new thread
            if self.current_index < len(self.patterns_list):
                threading.Thread(target=self.process_and_download).start()

            cl.process_dataset(dataset)
            upload_dataset(dataset)
            delete_dataset(filepath)


handler = DatasetHandler(allow_patterns_list)
handler.process_and_download()
