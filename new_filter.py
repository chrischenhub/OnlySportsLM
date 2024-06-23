import pandas as pd


keywords = [
    "football", "soccer", "basketball", "baseball", "tennis", 'athlete', 'running', 'marathon', 'copa', 'new', 'nike',
    'adidas'
    "cricket", "rugby", "golf", "volleyball", "sports", "sport", 'Sport', 'wrestling', 'wwe', 'hockey', 'volleyball',
    'cycling', 'swim',
    "athletic", "league", "team", "champion", "playoff", "olympic", 'premierleague', 'laliga', 'bundesliga', 'seriea',
    'ligue1', 'epl', 'laliga', 'bundesliga', 'seriea', 'ligue1', 'racing', 'nascar', 'motogp',
    "cup", "worldcup", "fitness", "workout", "gym", "nfl", "nba", 'NBA', 'NFL', 'MLB', 'NHL', 'FIFA', 'UEFA', 'NCAA',
    'MMA', 'UFC', 'ufc',
    "mlb", "nhl", "fifa", "uefa", "ncaa", 'boxing', 'espn', 'bleacherreport', 'mma', 'si.com', 'formula1', 'f1',
    'nytimes/athletic', 'apnews.com', 'goal',]

def filter_urls(df):
    sports_filter = [url for url in df['url'] if any(keyword in url for keyword in keywords)]
    df_filtered = df.loc[df['url'].isin(sports_filter)]
    return df_filtered


import os
from pyarrow import parquet as pq
from pyarrow import Table

input_directory = './downloads/test/data/CC-MAIN-2017-17/'
output_directory = './downloads/test/data/CC-MAIN-2017-17-filtered/'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for filename in os.listdir(input_directory):
    if filename.endswith(".parquet"):
        filepath = os.path.join(input_directory, filename)
        df = pd.read_parquet(filepath)
        filtered_df = filter_urls(df)
        output_path = os.path.join(output_directory, f"filtered_{filename}")
        filtered_df.to_parquet(output_path)
        del df, filtered_df
        os.remove(filepath)

import pyarrow as pa

def concatenate_parquet_files(input_directory, output_file):
    all_dfs = []
    for filename in os.listdir(input_directory):
        if filename.endswith(".parquet"):
            filepath = os.path.join(input_directory, filename)
            table = pq.read_table(filepath)
            all_dfs.append(table)
           # os.remove(filepath)  # Delete the filtered file after reading
            del table

    concatenated_table = pa.concat_tables(all_dfs)
    pq.write_table(concatenated_table, output_file)
    del all_dfs, concatenated_table

output_file = './downloads/outcome/CC-MAIN-2017-17'
concatenate_parquet_files(output_directory, output_file)

