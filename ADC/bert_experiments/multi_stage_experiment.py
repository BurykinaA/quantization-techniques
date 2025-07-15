import argparse
import torch
from torch.optim import AdamW
from transformers import get_scheduler, AutoModelForQuestionAnswering
from tqdm.auto import tqdm
import os
import time
import json
import matplotlib.pyplot as plt

from ADC.bert_experiments.bert_experiment_setup import get_squad_dataloaders
from ADC.bert_experiments.bert_evaluate import evaluate_model
from ADC.bert_experiments.quantized_bert_layers import adapt_model_for_stage
# Импортируем наши классы квантизаторов, чтобы проверять тип слоя
from ADC.quantizers import AffineQuantizerPerTensor, SymmetricQuantizerPerTensor

def set_quantizer_state(module, enabled: bool):
    """
    Recursively enables or disables observers in quantizer modules.
    """
    for child in module.children():
        if isinstance(child, (AffineQuantizerPerTensor, SymmetricQuantizerPerTensor)):
            if enabled:
                child.enable()
            else:
                child.disable()
        else:
            set_quantizer_state(child, enabled)


def train_one_stage(stage_name, model, train_dataloader, eval_dataloader, eval_examples, eval_features, device, args):
    print(f"\n----- Starting Stage: {stage_name} -----")
    
    epochs = args.qat_epochs if stage_name == 'QAT' else args.adc_epochs
    lr = args.qat_lr if stage_name == 'QAT' else args.adc_lr
    
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_dataloader)
    
    # Вычисляем, когда закончить калибровку (например, 20% от всех шагов)
    calibration_steps = int(num_training_steps * 0.2)
    print(f"Calibration will run for {calibration_steps} steps.")

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps), desc=f"{stage_name} Training")
    
    # Убедимся, что наблюдатели включены в начале
    set_quantizer_state(model, enabled=True)
    
    step_count = 0
    for epoch in range(epochs):
        model.train()
        for batch in train_dataloader:
            # Проверяем, не пора ли заморозить наблюдателей
            if step_count == calibration_steps:
                print(f"\n--- Step {step_count}: Calibration finished. Freezing observers. ---")
                model.apply(lambda m: m.disable() if hasattr(m, 'disable') else None)

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_description(f"Epoch {epoch+1}/{epochs} ({stage_name}), Loss: {loss.item():.4f}")
            step_count += 1

    print(f"----- Stage {stage_name} Finished -----")
    print("Evaluating model after stage...")
    metrics = evaluate_model(model, eval_dataloader, eval_examples, eval_features, device)
    print(f"Metrics after {stage_name}: {metrics}")
    return model, metrics

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"BERT_multistage_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Using device: {device}")
    print(f"Results will be saved to: {output_dir}")

    # Data
    train_dataloader, eval_dataloader, eval_examples, eval_features = get_squad_dataloaders(args.batch_size, args.subset_size)
    
    # --- STAGE 0: Load Initial FP Model on CPU ---
    fp_model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
    
    # --- STAGE 1: Quantization-Aware Training (QAT) ---
    print("Preparing model for QAT stage...")
    # Adapt model on CPU
    qat_model = adapt_model_for_stage(fp_model, 'qat', bw=args.bw, bx=args.bx, ba=args.ba, k=args.k)
    # Move the entire adapted model to the target device BEFORE training
    qat_model.to(device)
    
    qat_model, qat_metrics = train_one_stage("QAT", qat_model, train_dataloader, eval_dataloader, eval_examples, eval_features, device, args)
    
    # Сохраняем модель после QAT
    qat_model_path = os.path.join(output_dir, "qat_model.pt")
    torch.save(qat_model.state_dict(), qat_model_path)
    print(f"QAT model saved to {qat_model_path}")

    # --- STAGE 2: ADC Fine-tuning (RAOQ) ---
    print("\nPreparing model for ADC stage...")
    # 1. Загружаем веса QAT-модели. map_location=device гарантирует, что веса загрузятся сразу на нужное устройство.
    # Сначала создаем "скелет" модели с той же структурой, что и была сохранена (т.е. QAT-структурой)
    adc_model_base_skeleton = adapt_model_for_stage(
        AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased"),
        'qat', bw=args.bw, bx=args.bx, ba=args.ba, k=args.k
    )
    # Теперь загружаем веса в этот скелет
    adc_model_base_skeleton.load_state_dict(torch.load(qat_model_path, map_location=device))
    adc_model_base_skeleton.to(device) # Убедимся, что модель на нужном устройстве
    
    # 2. Теперь, когда у нас есть обученная QAT-модель на GPU, мы заменяем её QAT-слои на ADC-слои.
    # Новые ADC-слои будут созданы на CPU, поэтому после этого шага модель будет "гибридной".
    adc_model = adapt_model_for_stage(adc_model_base_skeleton, 'adc', bw=args.bw, bx=args.bx, ba=args.ba, k=args.k)
    
    # 3. Ключевой шаг: Снова перемещаем всю модель на целевое устройство, чтобы синхронизировать слои.
    adc_model.to(device)

    adc_model, final_metrics = train_one_stage("ADC", adc_model, train_dataloader, eval_dataloader, eval_examples, eval_features, device, args)

    # --- Save final results ---
    final_model_path = os.path.join(output_dir, "final_adc_model.pt")
    torch.save(adc_model.state_dict(), final_model_path)
    print(f"Final ADC model saved to {final_model_path}")

    results_payload = {
        'args': vars(args),
        'qat_metrics': qat_metrics,
        'final_metrics': final_metrics
    }
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(results_payload, f, indent=4)
    print("Final results saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT Multi-Stage Quantization Finetuning")
    parser.add_argument("--output_dir", type=str, default="bert_models_multistage", help="Directory to save models and results")
    
    # Quantization params
    parser.add_argument("--bx", type=int, default=4, help="Bit-width for activations")
    parser.add_argument("--bw", type=int, default=4, help="Bit-width for weights")
    parser.add_argument("--ba", type=int, default=8, help="Bit-width for ADC")
    parser.add_argument("--k", type=int, default=4, help="k-factor for ADC")

    # Training params
    parser.add_argument("--qat_epochs", type=int, default=2, help="Number of QAT epochs")
    parser.add_argument("--adc_epochs", type=int, default=2, help="Number of ADC finetuning epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--qat_lr", type=float, default=2e-5, help="Learning rate for QAT stage")
    parser.add_argument("--adc_lr", type=float, default=1e-5, help="Learning rate for ADC stage")
    parser.add_argument("--subset_size", type=int, default=1000, help="Use a subset of the dataset for quick testing")

    args = parser.parse_args()
    main(args) 