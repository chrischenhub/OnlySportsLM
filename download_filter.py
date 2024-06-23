import os
import pandas as pd
import glob
import argparse


def filter_sports_urls(df):
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

    sports_filter = [url for url in df['url'] if any(keyword in url for keyword in keywords)]
    df_filtered = df.loc[df['url'].isin(sports_filter)]
    return df_filtered


def process_parquet_files(directory):
    # 获取所有的.parquet文件
    parquet_files = glob.glob(os.path.join(directory, '*.parquet'))

    # 将文件等分为两组
    mid_point = (len(parquet_files) + 1) // 2
    group1_files = parquet_files[:mid_point]
    group2_files = parquet_files[mid_point:]

    # 处理第一组文件
    df1 = pd.concat([pd.read_parquet(file) for file in group1_files], ignore_index=True)
    df1_filtered = filter_sports_urls(df1)
    df1_filtered.to_parquet(os.path.join(directory, '1.parquet'))

    # 删除第一组原文件
    for file in group1_files:
        os.remove(file)

    # 处理第二组文件
    if group2_files:  # 如果第二组不为空
        df2 = pd.concat([pd.read_parquet(file) for file in group2_files], ignore_index=True)
        df2_filtered = filter_sports_urls(df2)
        df2_filtered.to_parquet(os.path.join(directory, '2.parquet'))

        # 删除第二组原文件
        for file in group2_files:
            os.remove(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process parquet files to filter sports URLs.")
    parser.add_argument("directory_path", type=str, help="Path to the directory containing parquet files.")
    args = parser.parse_args()

    process_parquet_files(args.directory_path)
