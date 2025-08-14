import argparse
import os
import time
import collections
import numpy as np
import torch
import evaluate
import torch
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


def prepare_train_features(examples, tokenizer, max_length=384, doc_stride=128):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offsets_mapping = tokenized.pop("offset_mapping")

    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    for i, offsets in enumerate(offsets_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            tokenized["start_positions"].append(cls_index)
            tokenized["end_positions"].append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        # context is always the second sequence
        context_index = 1

        # find token span of the context
        token_start_index = 0
        while sequence_ids[token_start_index] != context_index:
            token_start_index += 1
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != context_index:
            token_end_index -= 1

        # if answer not fully inside the context for this window -> CLS
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized["start_positions"].append(cls_index)
            tokenized["end_positions"].append(cls_index)
        else:
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char and sequence_ids[token_start_index] == context_index:
                token_start_index += 1
            start_position = token_start_index - 1

            while offsets[token_end_index][1] >= end_char and sequence_ids[token_end_index] == context_index:
                token_end_index -= 1
            end_position = token_end_index + 1

            tokenized["start_positions"].append(start_position)
            tokenized["end_positions"].append(end_position)

    return tokenized


def prepare_validation_features(examples, tokenizer, max_length=384, doc_stride=128):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    tokenized["example_id"] = []

    for i in range(len(tokenized["input_ids"])):
        sequence_ids = tokenized.sequence_ids(i)
        context_index = 1

        tokenized["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized["offset_mapping"][i])
        ]

        sample_index = sample_mapping[i]
        tokenized["example_id"].append(examples["id"][sample_index])

    return tokenized


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


def plot_curves(log_history, out_dir):
    import pandas as pd

    df = pd.DataFrame(log_history)

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


# def compute_metrics(eval_pred, eval_examples, eval_dataset, tokenizer, squad_metric):
#     """Compute metrics during training"""
#     try:
#         predictions, _ = eval_pred
#         print(f"DEBUG: Predictions shape: {predictions[0].shape if len(predictions) > 0 else 'No predictions'}")
        
#         # Postprocess predictions
#         formatted_predictions = postprocess_qa_predictions(
#             examples=eval_examples,
#             features=eval_dataset,
#             predictions=predictions,
#         )
#         print(f"DEBUG: Formatted {len(formatted_predictions)} predictions")
        
#         # Format for metric computation
#         references = [{"id": ex_id, "answers": ans} for ex_id, ans in zip(eval_examples["id"], eval_examples["answers"])]
#         predictions_for_metric = [{"id": k, "prediction_text": v} for k, v in formatted_predictions.items()]
        
#         # Compute SQuAD metrics
#         result = squad_metric.compute(predictions=predictions_for_metric, references=references)
#         print(f"DEBUG: SQuAD metrics computed: {result}")
        
#         # Return only the metrics that Trainer expects
#         return {"f1": result["f1"], "exact_match": result["exact_match"]}
        
#     except Exception as e:
#         print(f"ERROR in compute_metrics: {e}")
#         import traceback
#         traceback.print_exc()
#         # Return dummy metrics to avoid crash
#         return {"f1": 0.0, "exact_match": 0.0}

def compute_metrics(eval_pred):
    """Compute metrics during training - simplified version"""
    print("DEBUG: compute_metrics called!")
    
    # Just return dummy metrics to test if function is working
    return {
        "f1": 50.0,
        "exact_match": 30.0
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./outputs_qa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--eval_steps", type=int, default=2, help="Number of steps between evaluations")
    parser.add_argument("--save_steps", type=int, default=500, help="Number of steps between saves")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"squad_qa_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    raw = load_dataset("squad")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.padding_side = "right"
    model = BertForQuestionAnswering.from_pretrained(args.model_name)

    train_dataset = raw["train"].map(
        lambda x: prepare_train_features(x, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing train",
    )
    eval_examples = raw["validation"]
    eval_dataset = eval_examples.map(
        lambda x: prepare_validation_features(x, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=eval_examples.column_names,
        desc="Tokenizing validation",
    )
    
    print(f"DEBUG: eval_dataset created with {len(eval_dataset)} samples")
    print(f"DEBUG: eval_dataset columns: {eval_dataset.column_names}")

    metric = evaluate.load("squad")

    # Minimal TrainingArguments with compatibility fallback
    try:
        training_args = TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_train_epochs=args.num_train_epochs,
            warmup_ratio=args.warmup_ratio,
            logging_steps=50,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=2,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            fp16=args.fp16,
            report_to="none",
        )
    except TypeError:
        training_args = TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_train_epochs=args.num_train_epochs,
            warmup_steps=0,
            logging_steps=50,
            save_steps=args.save_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            fp16=args.fp16,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("DEBUG: Trainer created")
    print(f"DEBUG: Trainer has compute_metrics: {trainer.compute_metrics is not None}")
    
    # Test the compute_metrics function directly
    print("DEBUG: Testing compute_metrics function directly...")
    dummy_pred = (torch.randn(10, 384, 2), None)
    test_result = compute_metrics(dummy_pred)
    print(f"DEBUG: Direct test result: {test_result}")

    trainer.train()

    preds = trainer.predict(eval_dataset).predictions
    formatted = postprocess_qa_predictions(
        examples=eval_examples,
        features=eval_dataset,
        predictions=preds,
    )
    refs = [{"id": ex_id, "answers": ans} for ex_id, ans in zip(eval_examples["id"], eval_examples["answers"])]
    preds_for_metric = [{"id": k, "prediction_text": v} for k, v in formatted.items()]
    eval_metrics = metric.compute(predictions=preds_for_metric, references=refs)

    try:
        trainer.state.log_history.append({"step": trainer.state.global_step, "eval_f1": eval_metrics.get("f1", None)})
    except Exception:
        pass

    with open(os.path.join(out_dir, "eval_metrics.txt"), "w") as f:
        for k, v in sorted(eval_metrics.items()):
            f.write(f"{k}: {v}\n")

    plot_curves(trainer.state.log_history, out_dir)

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