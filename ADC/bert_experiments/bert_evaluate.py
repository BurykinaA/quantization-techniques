import torch
import collections
import numpy as np
from tqdm.auto import tqdm
import evaluate as evaluate_metric
import copy

def postprocess_qa_predictions(examples, features, raw_predictions, tokenizer, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = collections.OrderedDict()

    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]

        min_null_score = None
        valid_answers = []
        
        context = example["context"]
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
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
            best_answer = {"text": "", "score": 0.0}
        
        predictions[example["id"]] = best_answer["text"]

    return predictions

def evaluate_model(model, dataloader, device, tokenizer, all_examples, all_features):
    # It's important to have the model on the CPU before conversion
    # as some quantization ops might not be supported on CUDA.
    # We also create a copy to avoid modifying the original model.
    model_for_eval = copy.deepcopy(model)
    model_for_eval.to('cpu')
    model_for_eval.eval()

    # This is the crucial step to convert the QAT model to a truly quantized model.
    torch.quantization.convert(model_for_eval, inplace=True)
    
    model_for_eval.to(device)

    all_results = []
    for batch in tqdm(dataloader, desc="Evaluating"):
        model_inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']}
        
        with torch.no_grad():
            outputs = model_for_eval(**model_inputs)

        feature_indices = batch['overflow_to_sample_mapping']
        
        start_logits = outputs.start_logits.cpu().numpy()
        end_logits = outputs.end_logits.cpu().numpy()
        
        all_results.append((feature_indices.cpu().numpy(), start_logits, end_logits))

    metrics = score_model(all_examples, all_features, all_results, tokenizer)
    return metrics 