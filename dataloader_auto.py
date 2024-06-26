import argparse
import threading
import json
import time
import psutil
import os
import concurrent.futures
from main import allow_patterns_prefix, default_patterns_list, download_dataset, upload_dataset, delete_dataset, local_download_dir
from DataGenerator import keywords
from datasets import load_dataset, disable_caching
from filelock import FileLock
import subprocess



access_token = "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"
RETRY_LIMIT = 8  # 设置重试次数
DOWNLOAD_TIMEOUT = 300  # 设置下载超时时间（秒）
cache_dir = '/root/.cache/huggingface/'
uploaded_patterns_file = "dataloader_uploaded.txt"


# 禁用缓存
disable_caching()



def load_uploaded_patterns():
    if os.path.exists(uploaded_patterns_file):
        with open(uploaded_patterns_file, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_uploaded_pattern(pattern):
    with open(uploaded_patterns_file, 'a') as f:
        f.write(pattern + '\n')


def get_dir_size(dir_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def monitor_cache_dir(stop_event, size_change_event):
    initial_size = get_dir_size(cache_dir)
    while not stop_event.is_set():
        time.sleep(5)  # 每5秒检查一次
        current_size = get_dir_size(cache_dir)
        if current_size != initial_size:
            initial_size = current_size
            size_change_event.set()  # 大小改变，触发事件
        else:
            size_change_event.clear()  # 大小未改变，清除事件

def load_dataset_with_retry(name):
    retry_count = 0
    while retry_count < RETRY_LIMIT:
        try:
            stop_event = threading.Event()
            size_change_event = threading.Event()

            # 启动监控线程
            monitor_thread = threading.Thread(target=monitor_cache_dir, args=(stop_event, size_change_event))
            monitor_thread.start()

            def dataset_loader():
                return load_dataset("HuggingFaceFW/fineweb", name, split="train", num_proc=8)

            # 创建一个线程来加载数据集
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(dataset_loader)
                start_time = time.time()
                while True:
                    if future.done():
                        dataset = future.result()
                        break
                    if time.time() - start_time > DOWNLOAD_TIMEOUT and not size_change_event.is_set():
                        raise TimeoutError("Dataset loading stuck for 60 seconds without any disk usage change.")
                    time.sleep(1)

            stop_event.set()  # 停止监控线程
            monitor_thread.join()

            return dataset  # 加载成功，返回数据集
        except Exception as e:
            retry_count += 1
            wait_time = 5  # 等待时间
            print(f"Failed to load dataset, retrying after {wait_time} seconds... (Attempt {retry_count}/{RETRY_LIMIT}). Error: {str(e)}")
            time.sleep(wait_time)  # 等待指定时间后重试

            if retry_count >= RETRY_LIMIT:
                error_message = f"Failed to load dataset after {RETRY_LIMIT} retries. Error: {str(e)}"
                with open("load_error.txt", "a") as file:
                    file.write(error_message + "\n")
                print(error_message)
                return None  # 加载失败，返回None

def process_data(name):
    uploaded_patterns = load_uploaded_patterns()
    if name in uploaded_patterns:
        print(f"Pattern {name} has already been processed. Skipping.")
        return

    dataset = load_dataset_with_retry(name)
    if dataset is None:
        return  # 如果加载失败，则退出当前函数

    dataset = dataset.select_columns(['text', 'url', 'token_count'])
    print('Dataset loaded, filtering...')
    dataset = dataset.filter(lambda example: any(keyword in example["url"] for keyword in keywords), num_proc=8)
    print('Dataset filtered, uploading...')

    retry_count = 0
    while retry_count < RETRY_LIMIT:
        try:
            dataset.push_to_hub('Chrisneverdie/OnlySports', data_dir=name, private=False, max_shard_size="4096MB", token=access_token)
            print('Upload successful')
            break
        except Exception as e:
            retry_count += 1
            wait_time = 5 * 2 ** (retry_count - 1)  # 计算等待时间
            print(f"Upload failed, retrying after {wait_time} seconds... (Attempt {retry_count}/{RETRY_LIMIT})Error: {str(e)}")
            time.sleep(wait_time)  # 等待指定时间后重试

            if retry_count >= RETRY_LIMIT:
                error_message = f"Failed to upload dataset after {RETRY_LIMIT} retries. Error: {str(e)}"
                with open("upload_error.txt", "a") as file:
                    file.write(error_message + "\n")
                print(error_message)

    print('done')
    save_uploaded_pattern(name)  # 更新已处理的 pattern

def clear_cache():
    # 清除缓存的代码
    os.system('rm -rf ' + cache_dir + '*')

if __name__ == "__main__":
    # 环境变量增加下载速度
    command = "export HF_HUB_ENABLE_HF_TRANSFER=1"
    subprocess.run(command, shell=True, check=True)

    parser = argparse.ArgumentParser(description="Process parquet files to filter sports URLs.")
    parser.add_argument("-j", "--json", type=str, help="Path to the JSON file containing patterns")
    args = parser.parse_args()

    if args.json:
        with open(args.json, 'r') as f:
            data = json.load(f)
            patterns = data.get("patterns", [])

            for pattern in patterns:
                process_data(pattern)
                clear_cache()  # 在处理每个模式后清除缓存
    else:
        print("Please provide a JSON file containing the patterns.")