import os
import pandas as pd
import glob
import argparse
from huggingface_hub import snapshot_download

download_hub = "HuggingFaceFW/fineweb"
upload_hub = "Chrisneverdie/OnlySports"
local_download_dir = "./downloads/test"
access_token = "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"
allow_patterns_prefix = "data/"

keywords = ["football", "soccer", "basketball", "baseball", "tennis", 'athlete', 'running', 'marathon', 'copa',
            'new', 'nike', 'adidas',
            "cricket", "rugby", "golf", "volleyball", "sports", "sport", 'Sport', 'wrestling', 'wwe', 'hockey',
            'volleyball', 'cycling', 'swim',
            "athletic", "league", "team", "champion", "playoff", "olympic", 'premierleague', 'laliga', 'bundesliga',
            'seriea', 'ligue1', 'epl',
            'laliga', 'bundesliga', 'seriea', 'ligue1', 'racing', 'nascar', 'motogp', "cup", "worldcup", "fitness",
            "workout", "gym", "nfl",
            "nba", 'NBA', 'NFL', 'MLB', 'NHL', 'FIFA', 'UEFA', 'NCAA', 'MMA', 'UFC', 'ufc', "mlb", "nhl", "fifa",
            "uefa", "ncaa", 'boxing',
            'espn', 'bleacherreport', 'mma', 'si.com', 'formula1', 'f1', 'nytimes/athletic', 'apnews.com', 'goal']

def download_dataset(allow_patterns):
    filepath = snapshot_download(
        download_hub,
        repo_type="dataset",
        local_dir=local_download_dir,
        allow_patterns=allow_patterns_prefix + allow_patterns + "/*")
    print(f"Downloaded dataset to: {filepath}")
    return filepath

def filter_sports_urls(df):
    sports_filter = [url for url in df['url'] if any(keyword in url for keyword in keywords)]
    df_filtered = df.loc[df['url'].isin(sports_filter)]
    return df_filtered

def process_parquet_files(pattern, group_num):
    directory = download_dataset(pattern)
    print(f"Processing files in directory: {directory}")
    # 获取所有的.parquet文件
    parquet_files = glob.glob(os.path.join(directory, '*.parquet'))
    print(f"Found {len(parquet_files)} parquet files.")

    for i in range(0, len(parquet_files), group_num):
        group_files = parquet_files[i:i + group_num]
        print(f"Processing group of {len(group_files)} files starting from {group_files[0]} to {group_files[-1]}")

        if group_files:
            # 合并文件
            df_list = [pd.read_parquet(file) for file in group_files]
            df_merged = pd.concat(df_list, ignore_index=True)
            print("Files merged successfully.")

            # 过滤体育相关的URL
            df_filtered = filter_sports_urls(df_merged)
            print("URLs filtered successfully.")

            # 确定输出文件名
            first_file_name = os.path.basename(group_files[0]).rstrip(".parquet")
            last_file_name = os.path.basename(group_files[-1]).rstrip(".parquet")
            output_file_name = os.path.join(directory, f"{first_file_name}_{last_file_name}")
            print(f"Output file name: {output_file_name}")

            # 保存合并和过滤后的文件
            df_filtered.to_parquet(output_file_name)
            print("Output file saved successfully.")

            # 删除源文件
            for file in group_files:
                os.remove(file)
                print(f"Deleted file: {file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process parquet files to filter sports URLs.")
    parser.add_argument("-p", "--pattern", type=str, help="Target pattern in the hub", default="default_pattern")
    parser.add_argument("-g", "--group_num", type=int, help="Number of files to process in a group", default=10)
    args = parser.parse_args()

    process_parquet_files(args.pattern, args.group_num)
