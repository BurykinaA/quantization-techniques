# BERT QA Experiments

This directory contains scripts to train and evaluate BERT models for Question Answering on the SQUAD 1.1 dataset, including quantized versions.

## Setup

Before running the experiments, please install the required libraries:

```bash
pip install transformers datasets scikit-learn tqdm
```

## Usage

The main script to run an experiment is `bert_experiment.py`. You can configure the model, quantization parameters, and training hyperparameters via command-line arguments.

**Example: Train a baseline BERT model**
```bash
python ADC/bert_experiments/bert_experiment.py --model_type baseline --output_dir ./bert_baseline_model
```

**Example: Train a quantized BERT-ADC model**
```bash
python ADC/bert_experiments/bert_experiment.py --model_type adc --bx 8 --bw 4 --output_dir ./bert_adc_b8w4_model
```

Here is the experiment setup file, which handles data loading, preprocessing, and model creation.

```python:ADC/bert_experiments/bert_experiment_setup.py
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
)
from datasets import load_dataset

from ADC.quantized_layers import LinearADC, LinearADCAshift

MODEL_CHECKPOINT = "bert-base-uncased"
MAX_LENGTH = 384  # The maximum length of a feature (question and context)
DOC_STRIDE = 128  # The authorized overlap between two part of the context when splitting it is needed.

def _replace_linear(module, bx, bw, ba, k, ashift, name_prefix=""):
    for name, child in module.named_children():
        full_name = f"{name_prefix}.{name}" if name_prefix else name
        if isinstance(child, torch.nn.Linear):
            if "qa_outputs" in full_name or "attention" in full_name or "intermediate" in full_name:
                # Decide which quantized layer to use based on config
                QuantLayer = LinearADCAshift if ashift else LinearADC
                new_layer = QuantLayer(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    bx=bx,
                    bw=bw,
                    ba=ba,
                    k=k,
                    ashift=ashift,
                )
                setattr(module, name, new_layer)
        else:
            _replace_linear(child, bx, bw, ba, k, ashift, name_prefix=full_name)

def get_model(model_type, bx=8, bw=8, ba=8, k=4, ashift=False):
    config = AutoConfig.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT, config=config)

    if model_type == 'baseline':
        return model
    
    if model_type in ['adc', 'adc_ashift']:
        _replace_linear(model, bx=bx, bw=bw, ba=ba, k=k, ashift=ashift)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model

def get_squad_dataloaders(batch_size=16):
    datasets = load_dataset("squad")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # The preprocessing function needs to be applied in a batched way to leverage the tokenizer's speed
    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]

        # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features, each of length max_length, it's a sliding window approach.
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=MAX_LENGTH,
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature
        # to its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last token of the context if the answer is the last word
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    tokenized_datasets = datasets.map(prepare_train_features, batched=True, remove_columns=datasets["train"].column_names)
    
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size)
    
    # For evaluation, we need the raw validation set features
    validation_features = datasets["validation"].map(
        prepare_train_features,
        batched=True,
        remove_columns=datasets["validation"].column_names
    )

    return train_dataloader, eval_dataloader, datasets["validation"], validation_features
```

This is the evaluation script, which handles the SQUAD-specific post-processing and metrics calculation.

```python:ADC/bert_experiments/bert_evaluate.py
import torch
import collections
import numpy as np
from tqdm.auto import tqdm
from datasets import load_metric

def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map from a feature to its corresponding example.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []
        
        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our feature to text in the original context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(features[feature_index]["tokenizer"].cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the start and end position.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not found a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}
        
        predictions[example["id"]] = best_answer["text"]

    return predictions

def evaluate(model, dataloader, eval_examples, eval_features, device):
    model.eval()
    all_start_logits = []
    all_end_logits = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

        all_start_logits.append(start_logits.cpu().numpy())
        all_end_logits.append(end_logits.cpu().numpy())

    raw_predictions = (np.concatenate(all_start_logits), np.concatenate(all_end_logits))
    
    # The following part is CPU-intensive and depends on the tokenizer used.
    # We need to add the tokenizer to the features to make it accessible.
    # This is a bit of a hack.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    eval_features = eval_features.add_column("tokenizer", [tokenizer]*len(eval_features))
    eval_features = eval_features.add_column("example_id", eval_examples["id"])


    final_predictions = postprocess_qa_predictions(eval_examples, eval_features, raw_predictions)
    metric = load_metric("squad")
    
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]
    
    return metric.compute(predictions=formatted_predictions, references=references)
```

Finally, this is the main experiment script that ties everything together for training and evaluation.

```python:ADC/bert_experiments/bert_experiment.py
import argparse
import torch
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import os

from ADC.bert_experiments.bert_experiment_setup import get_model, get_squad_dataloaders
from ADC.bert_experiments.bert_evaluate import evaluate

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_dataloader, eval_dataloader, eval_examples, eval_features = get_squad_dataloaders(args.batch_size)
    
    # Model
    model = get_model(args.model_type, args.bx, args.bw, args.ba, args.k, args.ashift)
    model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # LR Scheduler
    num_training_steps = args.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Training loop
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(args.num_epochs):
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Evaluation
    print("Training finished. Starting evaluation...")
    metrics = evaluate(model, eval_dataloader, eval_examples, eval_features, device)
    print(f"Evaluation results: {metrics}")

    # Save model
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
        print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT SQUAD Experiment")
    parser.add_argument("--model_type", type=str, default="baseline", choices=["baseline", "adc", "adc_ashift"], help="Model type")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save model")
    
    # Quantization params
    parser.add_argument("--bx", type=int, default=8, help="Bit-width for activations")
    parser.add_argument("--bw", type=int, default=8, help="Bit-width for weights")
    parser.add_argument("--ba", type=int, default=8, help="Bit-width for ADC")
    parser.add_argument("--k", type=int, default=4, help="k-factor for ADC")
    parser.add_argument("--ashift", action="store_true", help="Enable A-Shift in ADC layers")

    # Training params
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")

    args = parser.parse_args()
    main(args) 