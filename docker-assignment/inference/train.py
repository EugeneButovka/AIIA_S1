import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainingArguments

DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "distilbert/distilbert-base-uncased-finetuned-sst-2-english")
DEFAULT_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR", "/models/finetuned-sst2")
DATASET_ID = "nyu-mll/glue"
DATASET_CONFIG = "sst2"
MAX_LENGTH = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on SST-2 for a few epochs")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_dataset(tokenizer: AutoTokenizer, samples: int, seed: int):
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="train")
    dataset = dataset.shuffle(seed=seed).select(range(min(samples, len(dataset))))

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=MAX_LENGTH)

    return dataset.map(tokenize, batched=True, remove_columns=["sentence", "idx"])


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_dataset = build_dataset(tokenizer, args.train_samples, args.seed)
    labels = load_dataset(DATASET_ID, DATASET_CONFIG, split="train").features["label"].names
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(labels))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"fine-tuned model saved to {args.output_dir}")


if __name__ == "__main__":
    main()