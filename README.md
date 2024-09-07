# OnlySportsLM

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Requirements

- Python 3.8+
- `huggingface_hub` library
- `transformers` library
- `datasets` library
- `torch` library
- `pandas` library
- `numpy` library

## Installation

1. Clone this repository:
   
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
## Usage
1. Download the target dataset to your local dir, preferably in parquet format.
   
2. Data Preprocessing
   2.1 Run sports_URL_filter.py on your download data folder. You can change the keywords list to your specific field. The processed file will be deleted and the filtered file will be saved to another dir.
   2.2 Run sports_classifier.py on your filtered data folder.
   
   To run the script, use the following command:
- `-t`, `--threads`: Number of threads to use in the thread pool. (Default: 3)
- `-j`, `--json`: Path to a JSON file containing a list of patterns to allow. You will include the folder names that contain parquet files in this JSON file (example provided below).
- `-n`, `--name`: Name of the folder with parquet files if --json is not provided.
  
   ### JSON File Structure
   
   The JSON file for custom patterns should have the following structure:
   
   ```json
   {
     "patterns": [
       "CC-MAIN-2013-20/000_00000.parquet",
       "CC-MAIN-2013-20/000_00001.parquet",
       ...
     ]
   }
   ```
   ```"CC-MAIN-2013-20/000_00000.parquet"```
   
   > process only 000_00000.parquet
   > 
   ```"CC-MAIN-2013-20"```
   > process all files in CC-MAIN-2013-20 as a batch

   ### Example:
   
   ```bash
   python sports_URL_filter.py -t 4 -j patterns.json
   python sports_classifier.py -n CC-MAIN-2013-20
   ```

3. Model Training
   For reference, use python 3.10, torch 2.3.1+cu121 (or latest), cuda 12.5+, latest deepspeed, but keep pytorch-lightning==1.9.5
   ```
   pip install torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu121
   pip install pytorch-lightning==1.9.5 deepspeed wandb ninja --upgrade
   
   cd model_training/
   ./demo-training-prepare.sh
   ./demo-training-run.sh
   (you may want to log in to wandb first)
      ```
   The current --my_exit_tokens number and --magic prime is set for training using [OnlySports Dataset (https://huggingface.co/datasets/Chrisneverdie/OnlySports_Dataset). Adjust if using another dataset.

4. Evaluation
   eval_question.jsonl contains 1000 generated sports prompts. Complete each prompt with your models, and combine and evaluate responses from different models using api_eval.ipynb. Feel free to add more LLMs to the notebook.
   Detailed performance of OnlySportsLM:
   
<img width="862" alt="image" src="https://github.com/user-attachments/assets/4f2ca9eb-965f-465c-994d-c5b79e68a528">

   


