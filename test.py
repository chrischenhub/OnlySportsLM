import os
import shutil

def delete_hf_datasets_folders():
    # 指定目标目录
    target_dir = 'tmp/'

    # 遍历目标目录下的所有文件和文件夹
    for root, dirs, files in os.walk(target_dir, topdown=False):
        for dir_name in dirs:
            if dir_name.startswith('hf_datasets'):
                # 构建完整路径
                full_path = os.path.join(root, dir_name)
                try:
                    # 删除文件夹及其内容
                    shutil.rmtree(full_path)
                    print(f"Deleted directory: {full_path}")
                except Exception as e:
                    print(f"Failed to delete {full_path}: {e}")

if __name__ == "__main__":
    delete_hf_datasets_folders()