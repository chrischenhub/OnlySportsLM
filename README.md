Ignore Readme for now, use [Huggingface](https://huggingface.co/collections/Chrisneverdie/onlysports-66b3e5cf595eb81220cc27a6) page for model and dataset.

# Dataset Handling Script

This project is designed to download datasets from Hugging Face, process them using a pre-trained classifier, and upload the processed datasets back to Hugging Face. It utilizes multi-threading to handle multiple patterns concurrently, ensuring efficient processing and uploading.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Script Explanation](#script-explanation)
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
2. 
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
## Usage
1. Download the target dataset to your local dir, preferably in parquet format.
   
2. Data preprocessing
   2.1 Run sports_URL_filter.py on your data dir. You can change the keywords list to your specific field. The processed file will be deleted and the filtered file will be saved to another dir.
   2.2 Run sports_classifier.py
   To run the script, use the following command:
- `-t`, `--threads`: Number of threads to use in the thread pool. (Default: 3)
- `-j`, `--json`: Path to a JSON file containing a list of patterns to allow. You will include the folder names that contain parquet files in this JSON file. (eg. If you are processing FineWeb, the downloaded files will have the following patterns as folder names:CC-MAIN-2013-20, CC-MAIN-2013-48...)

### Example:

```bash
python main.py -t 4 -j patterns.json
```

## Script Explanation

### main.py

This is the main script that handles the downloading, processing, and uploading of datasets.

- **download_dataset(allow_patterns)**: Downloads the dataset matching the allow pattern from Hugging Face.
- **upload_dataset(dataset)**: Uploads the processed dataset to Hugging Face.
- **delete_dataset(filepath)**: Deletes the dataset files from the local directory.
- **DatasetHandler**: Class that manages the downloading, processing, and uploading of datasets using multiple threads.
- **parse_args()**: Parses command-line arguments.
- **main()**: The main function that initializes the `DatasetHandler` and starts the processing.

### test_classifier.py

This script contains functions for loading and processing the dataset.

- **my_load_dataset(filepath)**: Loads the dataset from the specified file path.
- **compute_scores(batch)**: Computes the scores for each batch of texts using a pre-trained classifier.
- **add_prefix(example)**: Adds a prefix to each example based on computed scores.
- **process_dataset(dataset)**: Processes the entire dataset by computing scores and filtering based on a condition.

## JSON File Structure

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
