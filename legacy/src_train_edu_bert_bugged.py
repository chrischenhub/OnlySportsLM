from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
)
from datasets import load_dataset, ClassLabel
import numpy as np
import evaluate
import argparse
import os
from sklearn.metrics import classification_report, confusion_matrix


from sklearn.metrics import classification_report, confusion_matrix
import evaluate
import numpy as np
log_file_path = 'performance_log.txt'

def compute_metrics(eval_pred):
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    
    # Convert logits to binary predictions
    preds = np.argmax(logits, axis=1)
    labels = np.round(labels.squeeze()).astype(int)
    
    precision = precision_metric.compute(
        predictions=preds, references=labels, average="binary"
    )["precision"]
    recall = recall_metric.compute(
        predictions=preds, references=labels, average="binary"
    )["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="binary")["f1"]
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]

    report = classification_report(labels, preds)
    cm = confusion_matrix(labels, preds)
    print("Validation Report:\n" + report)
    print("Confusion Matrix:\n" + str(cm))

    with open(log_file_path, 'a') as log_file:
        log_file.write(report)
        log_file.write(str(cm))

        log_file.write("precision" + str(precision))
        log_file.write("recall" + str(recall))
        log_file.write("f1" + str(f1))
        log_file.write("accuracy" + str(accuracy))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }





def main(args):
    dataset = load_dataset('Chrisneverdie/sports-annotation',data_files={'train': 'train.parquet'})
    # dataset = load_dataset(
    #     "Chrisneverdie/sports-annotation", split="train", cache_dir="train/", num_proc=8
    # )
    dataset = dataset.map(
        lambda x: {args.target_column: np.clip(int(x[args.target_column]), 0, 5)}, num_proc=8
    )

    dataset = dataset.cast_column(
        args.target_column, ClassLabel(names=[str(i) for i in range(6)])
    )
    dataset = dataset['train'].train_test_split(
        train_size=0.9, seed=8, stratify_by_column=args.target_column
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

    def preprocess(examples):
        batch = tokenizer(examples["text"], truncation=True)
        batch["labels"] = np.float32(examples[args.target_column])
        return batch

    dataset = dataset.map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model_name, num_labels=2, classifier_dropout=0.0, hidden_dropout_prob=0.0)

    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for param in model.bert.encoder.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=100,
        learning_rate=3e-4,
        num_train_epochs=10,
        seed=8,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=128,
        load_best_model_at_end=True,
        #metric_for_best_model="f1_macro",
        greater_is_better=True,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(os.path.join(args.checkpoint_dir, "final"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default="Snowflake/snowflake-arctic-embed-xs")
    parser.add_argument("--dataset_name", type=str, default="Chrisneverdie/sports-annotation")
    parser.add_argument("--target_column", type=str, default="label")
    parser.add_argument("--checkpoint_dir", type=str, default="ckpt/")
    args = parser.parse_args()

    main(args)
