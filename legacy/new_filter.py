import pandas as pd
access_token = "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"

from pyarrow import parquet as pq
from pyarrow import Table

from datasets import load_dataset, Dataset, DatasetDict,concatenate_datasets
from huggingface_hub import snapshot_download
import argparse
import os
from pyarrow import parquet as pq
from pyarrow import Table
import gc

download_hub = "HuggingFaceFW/fineweb"
upload_hub = "Chrisneverdie/OnlySports"
local_download_dir = "./downloads/test"
access_token = "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"
allow_patterns_prefix = "data/"


keywords = [
    "football", "soccer", "basketball", "baseball", "tennis", 'athlete', 'running', 'marathon', 'copa', 'new', 'nike',
    'adidas', "cricket", "rugby", "golf", "volleyball", "sports", "sport", 'Sport', 'wrestling', 'wwe', 'hockey', 
    'volleyball', 'cycling', 'swim', "athletic", "league", "team", "champion", "playoff", "olympic", 'premierleague', 
    'laliga', 'bundesliga', 'seriea', 'ligue1', 'epl', 'racing', 'nascar', 'motogp', "cup", "worldcup", "fitness", 
    "workout", "gym", "nfl", "nba", 'NBA', 'NFL', 'MLB', 'NHL', 'FIFA', 'UEFA', 'NCAA', 'MMA', 'UFC', 'ufc', "mlb", 
    "nhl", "fifa", "uefa", "ncaa", 'boxing', 'espn', 'bleacherreport', 'mma', 'si.com', 'formula1', 'f1', 
    'nytimes/athletic', 'apnews.com', 'goal',
]


# def filter_urls(df):
#     sports_filter = [url for url in df['url'] if any(keyword in url for keyword in keywords)]
#     df_filtered = df.loc[df['url'].isin(sports_filter)]
#     return df_filtered

# def filter_urls(df):
#     pattern = '|'.join(keywords)  # Create a regex pattern from the keywords
#     df_filtered = df[df['url'].str.contains(pattern, case=False, na=False)]  # Use vectorized string operations
#     return df_filtered


def download_dataset(allow_patterns):
    filepath = snapshot_download(
        download_hub,
        repo_type="dataset",
        local_dir=local_download_dir,
        allow_patterns=allow_patterns_prefix + allow_patterns + "/*")
    print(f"Downloaded dataset to: {filepath}")
    return filepath

# for filename in os.listdir(input_directory):
#     if filename.endswith(".parquet"):
#         filepath = os.path.join(input_directory, filename)
#         df = pd.read_parquet(filepath)
#         filtered_df = filter_urls(df)
#         output_path = os.path.join(output_directory, f"filtered_{filename}")
#         filtered_df.to_parquet(output_path)
#         print(f'{filename} filtered')
#         del df, filtered_df
#         os.remove(filepath)
#         gc.collect()


# import pyarrow as pa

# def concatenate_parquet_files(input_directory, output_file):
#     all_dfs = []
#     for filename in os.listdir(input_directory):
#         if filename.endswith(".parquet"):
#             filepath = os.path.join(input_directory, filename)
#             table = pq.read_table(filepath)
#             all_dfs.append(table)
#            # os.remove(filepath)  # Delete the filtered file after reading
#             del table

#     concatenated_table = pa.concat_tables(all_dfs)
#     pq.write_table(concatenated_table, output_file)
#     del all_dfs, concatenated_table

# print('Concatenating filtered files...')
# output_file = './downloads/outcome/CC-MAIN-2017-17'
# concatenate_parquet_files(output_directory, output_file)



def filter_dataset(dataset, keywords):
    return dataset.filter(lambda example: any(keyword in example["url"] for keyword in keywords),num_proc=8)

def process_and_filter_files(patterns):
    directory = download_dataset(patterns)
    all_filtered_datasets = []
    for filename in os.listdir(directory):
        if filename.endswith(".parquet"):
            filepath = os.path.join(directory, filename)
            dataset = load_dataset('parquet', data_files=filepath, split='train',num_proc=8)
            dataset = dataset.select_columns(['text','url','token_count'])
            filtered_dataset = filter_dataset(dataset, keywords)
            #output_path = os.path.join(output_directory, f"filtered_{filename}")
            #filtered_dataset.to_parquet(output_path)
            all_filtered_datasets.append(filtered_dataset)
            os.remove(filepath) 
            gc.collect()
    concatenated_dataset = concatenate_datasets(all_filtered_datasets)
    #concatenated_dataset.to_parquet(final_output_file)
    #concatenated_dataset.push_to_hub(repo_id='repo_id', dataset_name=dataset_name)
    concatenated_dataset.push_to_hub("Chrisneverdie/OnlySports", data_dir=patterns, private=False, max_shard_size="4096MB", token=access_token)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process parquet files to filter sports URLs.")
    parser.add_argument("-n", "--name", type=str, help="Target pattern in the hub", default="default_pattern")
    args = parser.parse_args()

    process_and_filter_files(args.name)
