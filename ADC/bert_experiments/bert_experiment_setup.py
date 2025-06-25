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

def get_squad_dataloaders(batch_size=16, subset_size=None):
    datasets = load_dataset("squad")

    if subset_size is not None:
        print(f"Using a subset of the data: {subset_size} examples.")
        datasets["train"] = datasets["train"].select(range(subset_size))
        datasets["validation"] = datasets["validation"].select(range(subset_size))

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