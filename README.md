data_filter.py - filter out sports and non-sports dataset

test_classifier.py - run classification on HF dataset, output a dataset directly to HF with predicted labels.

src_train_edu_bert.py - train sports text classifier

src_run_edu_bert.py - templet for running text classifier


```
python DataGenerator.py --file_path './001_00000.parquet'--label 1--api_key 'your_api_key_here'--max_counter 20000
```
