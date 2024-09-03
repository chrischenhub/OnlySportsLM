import requests
from bs4 import BeautifulSoup
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 加锁
lock = threading.Lock()

# 主 URL
mainUrl = "https://huggingface.co/datasets/HuggingFaceFW/fineweb/tree/main/data"

def fetch_and_process(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    span_elements = soup.find_all('span', class_='truncate')

    results = []
    for span in span_elements:
        if span.string and span.string.startswith("CC-MAIN"):
            new_url = f"{mainUrl}/{span.string}"
            results.append((new_url, span.string))
    return results

def process_sub_page(sub_url, prefix):
    response = requests.get(sub_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    span_elements = soup.find_all('span', class_='truncate group-hover:underline')

    results = []
    for span in span_elements:
        if span.string:
            results.append(f"{prefix}/{span.string}")

    print("prefix:" + prefix + "  url: \n" + str(results))

    with lock:
        if not os.path.exists('result.json'):
            with open('result.json', 'w') as f:
                json.dump({"pattern_lists": []}, f)

        with open('result.json', 'r') as f:
            data = json.load(f)

        with open('result.json', 'w') as f:
            json.dump(data, f, indent=4)

def main():
    results = fetch_and_process(mainUrl)
    with ThreadPoolExecutor(max_workers=5) as executor:  # 设置线程池大小为5
        try:
            for sub_url, prefix in results:
                executor.submit(process_sub_page, sub_url, prefix)
        except Exception as exc:
            print(f"{sub_url} generated an exception: {exc}")



if __name__ == "__main__":
    main()
