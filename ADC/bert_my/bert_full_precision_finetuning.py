import argparse
import os
import time
import collections
import numpy as np
import torch
import evaluate
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BertForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import EvalPrediction


def prepare_train_features(examples, tokenizer, max_length=384, doc_stride=128):
    pad_on_right = tokenizer.padding_side == "right"

    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        # Find the start and end of the context in the tokenized sequence
        if pad_on_right:
            context_index = 1
        else:
            context_index = 0

        # Find the token start and end for the context
        token_start_index = 0
        while sequence_ids[token_start_index] != context_index:
            token_start_index += 1
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != context_index:
            token_end_index -= 1

        # If the answer is not fully inside the context, label CLS
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Move the token_start_index and token_end_index to the answer boundaries
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char and sequence_ids[token_start_index] == context_index:
                token_start_index += 1
            start_position = token_start_index - 1

            while offsets[token_end_index][1] >= end_char and sequence_ids[token_end_index] == context_index:
                token_end_index -= 1
            end_position = token_end_index + 1

            tokenized_examples["start_positions"].append(start_position)
            tokenized_examples["end_positions"].append(end_position)

    return tokenized_examples


def prepare_validation_features(examples, tokenizer, max_length=384, doc_stride=128):
    pad_on_right = tokenizer.padding_side == "right"

    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if (tokenizer.padding_side == "right") else 0

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

    return tokenized_examples


def postprocess_qa_predictions(
    examples,
    features,
    predictions,
    n_best_size=20,
    max_answer_length=30,
):
    all_start_logits, all_end_logits = predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feat_id in enumerate(features["example_id"]):
        features_per_example[feat_id].append(i)

    predictions_dict = {}

    for example_id, feature_indices in features_per_example.items():
        context = examples["context"][example_id_to_index[example_id]]

        min_null_score = None
        prelim_predictions = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features["offset_mapping"][feature_index]

            start_indexes = np.argsort(start_logits)[-1:-n_best_size-1:-1].tolist()
            end_indexes = np.argsort(end_logits)[-1:-n_best_size-1:-1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    prelim_predictions.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start": start_char,
                            "end": end_char,
                        }
                    )

        if len(prelim_predictions) == 0:
            predictions_dict[example_id] = ""
            continue

        best_pred = max(prelim_predictions, key=lambda x: x["score"])
        predictions_dict[example_id] = context[best_pred["start"]: best_pred["end"]]

    return predictions_dict


def build_post_processing_function(tokenizer, n_best_size=20, max_answer_length=30):
    def post_processing_function(examples, features, predictions, training_args):
        preds = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            n_best_size=n_best_size,
            max_answer_length=max_answer_length,
        )

        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in preds.items()]
        references = [{"id": ex_id, "answers": ans} for ex_id, ans in zip(examples["id"], examples["answers"])]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)
    return post_processing_function


def plot_curves(log_history, out_dir):
    import pandas as pd

    df = pd.DataFrame(log_history)

    # Training loss curve
    train_logs = df[df["loss"].notna()][["step", "loss"]].dropna()
    if not train_logs.empty:
        plt.figure(figsize=(7, 4))
        plt.plot(train_logs["step"], train_logs["loss"], label="train_loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=200)
        plt.close()

    # Eval F1 curve
    if "eval_f1" in df.columns:
        eval_logs = df[df["eval_f1"].notna()][["step", "eval_f1"]].dropna()
        if not eval_logs.empty:
            plt.figure(figsize=(7, 4))
            plt.plot(eval_logs["step"], eval_logs["eval_f1"], label="eval_f1")
            plt.xlabel("Step")
            plt.ylabel("F1")
            plt.title("Evaluation F1")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "f1_curve.png"), dpi=200)
            plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./outputs_qa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["epoch", "steps"])
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--report_to", type=str, default="none")
    args = parser.parse_args()

    set_seed(args.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"squad_qa_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    raw_datasets = load_dataset("squad")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = BertForQuestionAnswering.from_pretrained(args.model_name)

    # Preprocess
    train_dataset = raw_datasets["train"].map(
        lambda x: prepare_train_features(x, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing train dataset",
    )
    eval_examples = raw_datasets["validation"]
    eval_dataset = eval_examples.map(
        lambda x: prepare_validation_features(x, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=eval_examples.column_names,
        desc="Tokenizing validation dataset",
    )

    data_collator = default_data_collator
    metric = evaluate.load("squad")
    post_processing_function = build_post_processing_function(tokenizer)

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    training_args = TrainingArguments(
        output_dir=out_dir,
        evaluation_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=max(10, args.eval_steps // 10 if args.eval_strategy == "steps" else 50),
        save_strategy=args.eval_strategy,
        save_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=args.fp16,
        report_to=args.report_to.split(",") if args.report_to != "none" else "none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Evaluate with post-processing to get EM/F1
    eval_metrics = trainer.evaluate(
        eval_dataset=eval_dataset,
        eval_examples=eval_examples,
        metric_key_prefix="eval",
        # The Trainer will call post_process if provided via self.post_process_function attribute.
    )
    # Manually run post-processing + compute to ensure metrics (compat across versions)
    try:
        preds = trainer.predict(eval_dataset=eval_dataset, eval_examples=eval_examples)
        # If Trainer used post-processing internally, preds.metrics already includes f1/em
        if "test_f1" in preds.metrics or "eval_f1" in eval_metrics:
            pass
        else:
            # Build ourselves
            formatted = post_processing_function(eval_examples, eval_dataset, preds.predictions, training_args)
            metrics_final = metric.compute(predictions=formatted.predictions, references=formatted.label_ids)
            eval_metrics.update({f"eval_{k}": v for k, v in metrics_final.items()})
    except Exception:
        # Fallback path for older versions: run a minimal post-process
        logits = trainer.predict(eval_dataset).predictions
        formatted = post_processing_function(eval_examples, eval_dataset, logits, training_args)
        metrics_final = metric.compute(predictions=formatted.predictions, references=formatted.label_ids)
        eval_metrics.update({f"eval_{k}": v for k, v in metrics_final.items()})

    # Save metrics
    with open(os.path.join(out_dir, "eval_metrics.txt"), "w") as f:
        for k, v in sorted(eval_metrics.items()):
            f.write(f"{k}: {v}\n")

    # Plot curves
    plot_curves(trainer.state.log_history, out_dir)

    # Save model and tokenizer
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    print("Final metrics:", eval_metrics)
    print(f"Artifacts saved to: {out_dir}")
    if os.path.exists(os.path.join(out_dir, "loss_curve.png")):
        print(f"Loss curve: {os.path.join(out_dir, 'loss_curve.png')}")
    if os.path.exists(os.path.join(out_dir, "f1_curve.png")):
        print(f"F1 curve: {os.path.join(out_dir, 'f1_curve.png')}")


if __name__ == "__main__":
    main()