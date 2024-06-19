import pandas as pd
from openai import OpenAI
import argparse

parser = argparse.ArgumentParser(description='Process some data.')

parser.add_argument('--file_path', 
                    type=str, 
                    required=True, 
                    help='The path to the parquet file')

parser.add_argument('--label', 
                    type=int, 
                    required=True, 
                    help='The label to process, 0 for non-sports, 1 for sports')

parser.add_argument('--api_key', 
                    type=str, 
                    required=False, 
                    help='API key for OpenAI')

parser.add_argument('--max_counter', 
                    type=int, 
                    default=20000, 
                    help='Maximum number of items to process')

def main(FilePath, label, apiKey, MaxCounter):
    df = ReadParquet(FilePath)
    if label == 0:
        NonSports, WithLabel = KeywordsFilter(df, label) 
    elif label == 1:
        WithLabel = KeywordsFilter(df, label)


    client = OpenAI(
        base_url="https://av37dmiu0tjje1sr.us-east-1.aws.endpoints.huggingface.cloud/v1/", 
        api_key="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" ) 
        
    results_df = Checker(WithLabel, label, MaxCounter, client)
    #if label == 0: PostProcess0(NonSports, results_df) 
    if label == 1: PostProcess1(WithLabel, results_df)

def ReadParquet(FilePath):
    df = pd.read_parquet(FilePath)

    df = df[df['language'] == 'en'][['text','url','token_count']]

    return df

def KeywordsFilter(df, label):
    # keywords = [
    #     "football", "soccer", "basketball", "baseball", "tennis", 
    #     "cricket", "rugby", "golf", "volleyball", "sports", "sport", 
    #     "athletics", "league", "team", "champion", "playoff", "olympics",
    #     "cup", "worldcup", "fitness", "workout", "gym", "nfl", "nba", 
    #     "mlb", "nhl", "fifa", "uefa", "ncaa" 
    # ]

    keywords = [
    "football", "soccer", "basketball", "baseball", "tennis",'athlete','running','marathon','copa','new','nike', 'adidas', 
    "cricket", "rugby", "golf", "volleyball", "sports", "sport", 'Sport','wrestling','wwe', 'hockey','volleyball','cycling','swim',
    "athletic", "league", "team", "champion", "playoff", "olympic",'premierleague','laliga','bundesliga','seriea','ligue1','epl','laliga','bundesliga','seriea','ligue1','racing','nascar','motogp',
    "cup", "worldcup", "fitness", "workout", "gym", "nfl", "nba", 'NBA','NFL','MLB','NHL','FIFA','UEFA','NCAA','MMA','UFC','ufc',
    "mlb", "nhl", "fifa", "uefa", "ncaa",'boxing','espn','bleacherreport','mma','si.com','formula1','f1','nytimes/athletic','apnews.com','goal',
    ]
    
    NonSportsFilter = [url for url in df['url'] if not any(keyword in url for keyword in keywords)]
    SportsFilter = [url for url in df['url'] if any(keyword in url for keyword in keywords)]

    filter = NonSportsFilter if label == 0 else SportsFilter

    Labeled = df.loc[df['url'].isin(filter)]
    Labeled['label'] = label

    Labeled = Labeled[['text','token_count', 'label']].reset_index(drop=True)

    if label == 0:
        GroupKeys = Labeled.index // 10
        Grouped = Labeled.groupby(GroupKeys).sum()
        Grouped.reset_index(drop=True, inplace=True)

        return Labeled, Grouped
    
    elif label == 1:
        return Labeled
# If necessary, install the openai Python library by running 
# pip install openai

def SportsChecker(client, text):

    response = client.chat.completions.create(
        model="meta/llama3-8b-instruct",
        messages=[
            {"role": "system", 
             "content": """Given the following text, determine if any sentence in it is related to sports. 
             Consider a sentence as sports-related if it mentions sports stars, sports events, 
             sports news, general sports activities, etc.. Strictly respond with 'sports-related' 
             if it meets these criteria and 'not-sports-related' if it does not."""},  # context
            
            {"role": "user", 
             "content": f"Given the following text, determine if any sentence in it is related to sports. Text: {text}"}  # prompt
        ],
        temperature=0.1
    )
    return response.choices[0].message.content.strip()

def NonSportsChecker(client, text):
    response = client.chat.completions.create(
        model="meta/llama3-8b-instruct",
        messages=[
            {"role": "system", 
             "content": """Determine if the majority of the given text is related to sports. 
                           A text is considered 'sports-related' if more than 50 percent of its content 
                           discusses sports stars, sports events, sports news, or general sports activities. 
                           Respond with 'sports-related' if these criteria are met. If the text does not focus 
                           predominantly on sports, respond with 'not-sports-related'. Respond only with 
                           'sports-related' or 'not-sports-related'."""},  # context
            
            {"role": "user", 
             "content": f"Analyze this text to determine its relation to sports. Text: {text}"}  # prompt
        ],
        temperature=0.1
    )
    return response.choices[0].message.content.strip()


def Checker(WithLabel, label, MaxCounter, client):
    results = []

    counter = 0

    for index, row in WithLabel.iterrows():
        try:
            print(f"Currently working on index {index}...")
            if label == 0:
                continue
                #result = SportsChecker(client, row['text'])
            elif label == 1:
                result = NonSportsChecker(client, row['text'])
            results.append((index, result))
        except Exception as e:
            print(f"An error occurred at index {index}: {e}")
            results.append((index, None))  
        
        counter += 1
        if counter == MaxCounter:
            break

    results_df = pd.DataFrame(results, columns=['Index', 'Response'])

    return results_df

def PostProcess0(NonSports, results_df):
    NonSports['Group'] = NonSports.index // 10

    filtered_indices = results_df[results_df['Response'] == "not-sports-related"].index
    filtered_NonSports = NonSports[NonSports['Group'].isin(filtered_indices)]

    filtered_NonSports.drop(columns='Group', inplace=True)

    filtered_NonSports.to_parquet('./label0.parquet', index=False)

def PostProcess1(WithLabel, results_df):
    filtered_indices = results_df[results_df['Response'] == "sports-related"].index
    filtered_Sports = WithLabel.iloc[filtered_indices]

    filtered_Sports.to_parquet('./label1.parquet', index=False)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


