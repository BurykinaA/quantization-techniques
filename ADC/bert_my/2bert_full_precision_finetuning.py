import argparse
import os
import time
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import evaluate
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BertForQuestionAnswering,
    default_data_collator,
    set_seed,
    get_linear_schedule_with_warmup,
)


# Raw SQuAD dataset example structure:
# {
#     "id": "5733be284776f41900661182",
#     "title": "University_of_Notre_Dame", 
#     "context": "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms outstretched with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart...",
#     "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
#     "answers": {
#         "text": ["Saint Bernadette Soubirous"],
#         "answer_start": [515]  # Character position where answer starts in context
#     }
# }


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
    # tokenized = {
    # "input_ids": [[CLS, Where, does, John, live, ?, SEP, John, lives, in, Paris, ,, the, capital, of, France, ., SEP, PAD, PAD, ...]],
    # "attention_mask": [[1, 1, 1, ... , 0, 0, ...]],
    # "token_type_ids": [[0,0,0,0,0,0,0, 1,1,1,1,1,1,1,...]],
    # "offset_mapping": [[(0,0), (0,5), (6,10), ..., (13,18), ...]],
    # "overflow_to_sample_mapping": [0]
    # }

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


class MetricsComputer:
    def __init__(self, eval_examples, eval_dataset, tokenizer, squad_metric):
        self.eval_examples = eval_examples
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.squad_metric = squad_metric
    
    def compute_metrics(self, eval_pred):
        """Compute metrics during training"""
        try:
            predictions, _ = eval_pred
            
            # Postprocess predictions
            formatted_predictions = postprocess_qa_predictions(
                examples=self.eval_examples,
                features=self.eval_dataset,
                predictions=predictions,
            )
            
            # Format for metric computation
            references = [{"id": ex_id, "answers": ans} for ex_id, ans in zip(self.eval_examples["id"], self.eval_examples["answers"])]
            predictions_for_metric = [{"id": k, "prediction_text": v} for k, v in formatted_predictions.items()]
            
            # Compute SQuAD metrics
            result = self.squad_metric.compute(predictions=predictions_for_metric, references=references)
            
            return {"f1": result["f1"], "exact_match": result["exact_match"]}
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"f1": 0.0, "exact_match": 0.0}

def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Print progress
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1}/{total_epochs}, Batch {batch_idx}/{num_batches}, "
                  f"Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
    
    avg_loss = total_loss / num_batches
    return avg_loss

def evaluate_model(model, eval_loader, eval_examples, eval_dataset, tokenizer, metric, device):
    """Evaluate the model"""
    model.eval()
    all_start_logits = []
    all_end_logits = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                total_loss += outputs.loss.item()
            
            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())
    
    # Concatenate all predictions
    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)
    
    # Postprocess predictions
    formatted_predictions = postprocess_qa_predictions(
        examples=eval_examples,
        features=eval_dataset,
        predictions=(all_start_logits, all_end_logits),
    )
    
    # Format for metric computation
    references = [{"id": ex_id, "answers": ans} for ex_id, ans in zip(eval_examples["id"], eval_examples["answers"])]
    predictions_for_metric = [{"id": k, "prediction_text": v} for k, v in formatted_predictions.items()]
    
    # Compute SQuAD metrics
    result = metric.compute(predictions=predictions_for_metric, references=references)
    
    avg_loss = total_loss / len(eval_loader) if len(eval_loader) > 0 else 0.0
    
    return {
        "eval_loss": avg_loss,
        "eval_f1": result["f1"],
        "eval_exact_match": result["exact_match"]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./outputs_qa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--eval_steps", type=int, default=500, help="Number of steps between evaluations")
    parser.add_argument("--save_steps", type=int, default=1000, help="Number of steps between saves")
    args = parser.parse_args()

    set_seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"squad_qa_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # Load dataset and tokenizer
    raw = load_dataset("squad")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.padding_side = "right"
    model = BertForQuestionAnswering.from_pretrained(args.model_name)
    model.to(device)

    # Prepare datasets
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
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
        pin_memory=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
        pin_memory=True
    )
    
    metric = evaluate.load("squad")

    # Setup optimizer and scheduler
    num_training_steps = len(train_loader) * args.num_train_epochs
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Training loop
    log_history = []
    global_step = 0
    best_f1 = 0.0
    
    print(f"Starting training for {args.num_train_epochs} epochs...")
    print(f"Total training steps: {num_training_steps}")
    print(f"Warmup steps: {num_warmup_steps}")
    
    for epoch in range(args.num_train_epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.num_train_epochs} ===")
        
        # Training
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Log training loss
            if global_step % 50 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"Step {global_step}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}, "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")
                log_history.append({
                    "step": global_step,
                    "loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0]
                })
            
            # Evaluation
            if global_step % args.eval_steps == 0:
                print(f"\n--- Evaluation at step {global_step} ---")
                eval_metrics = evaluate_model(
                    model, eval_loader, eval_examples, eval_dataset, tokenizer, metric, device
                )
                
                print(f"Eval Loss: {eval_metrics['eval_loss']:.4f}")
                print(f"Eval F1: {eval_metrics['eval_f1']:.2f}")
                print(f"Eval Exact Match: {eval_metrics['eval_exact_match']:.2f}")
                
                # Log evaluation metrics
                log_entry = {
                    "step": global_step,
                    "eval_loss": eval_metrics['eval_loss'],
                    "eval_f1": eval_metrics['eval_f1'],
                    "eval_exact_match": eval_metrics['eval_exact_match']
                }
                log_history.append(log_entry)
                
                # Save best model
                if eval_metrics['eval_f1'] > best_f1:
                    best_f1 = eval_metrics['eval_f1']
                    best_model_path = os.path.join(out_dir, "best_model")
                    os.makedirs(best_model_path, exist_ok=True)
                    model.save_pretrained(best_model_path)
                    tokenizer.save_pretrained(best_model_path)
                    print(f"New best F1: {best_f1:.2f} - Model saved to {best_model_path}")
                
                model.train()  # Reset to training mode
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                checkpoint_path = os.path.join(out_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_path, exist_ok=True)
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_metrics = evaluate_model(
        model, eval_loader, eval_examples, eval_dataset, tokenizer, metric, device
    )
    
    print(f"Final Eval Loss: {final_metrics['eval_loss']:.4f}")
    print(f"Final F1 Score: {final_metrics['eval_f1']:.2f}")
    print(f"Final Exact Match: {final_metrics['eval_exact_match']:.2f}")

    # Save final model
    final_model_path = os.path.join(out_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    # Save metrics
    with open(os.path.join(out_dir, "eval_metrics.txt"), "w") as f:
        for k, v in sorted(final_metrics.items()):
            f.write(f"{k}: {v}\n")
        f.write(f"best_f1: {best_f1}\n")

    # Save training log
    import json
    with open(os.path.join(out_dir, "training_log.json"), "w") as f:
        json.dump(log_history, f, indent=2)

    # Plot curves
    plot_curves(log_history, out_dir)

    print(f"\nTraining completed!")
    print(f"Best F1 Score: {best_f1:.2f}")
    print(f"Final F1 Score: {final_metrics['eval_f1']:.2f}")
    print(f"All artifacts saved to: {out_dir}")
    if os.path.exists(os.path.join(out_dir, "loss_curve.png")):
        print(f"Loss curve: {os.path.join(out_dir, 'loss_curve.png')}")
    if os.path.exists(os.path.join(out_dir, "f1_curve.png")):
        print(f"F1 curve: {os.path.join(out_dir, 'f1_curve.png')}")


if __name__ == "__main__":
    main()