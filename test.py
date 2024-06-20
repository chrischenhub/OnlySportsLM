from main import local_download_dir, allow_patterns_prefix
import os

pattern = "pattern"


def process_file(file_path):
    print("loading file {}\n".format(file_path))
    dataset = cl.my_load_dataset(file_path)
    print("finished loading file {}, start filtering\n".format(file_path))
    dataset = dataset.select_columns(['text', 'url', 'dump', 'token_count'])
    dataset = dataset.filter(lambda example: any(keyword in example["url"] for keyword in keywords))
    print("finished filtering file {}, start uploading\n", format(file_path))



print(full_paths)
