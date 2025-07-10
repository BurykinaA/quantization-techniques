import argparse
import torch
from transformers import AutoModelForQuestionAnswering

from ADC.bert_experiments.bert_experiment_setup import get_squad_dataloaders
from ADC.bert_experiments.bert_evaluate import evaluate
from ADC.bert_experiments.quantized_bert_layers import adapt_model_for_stage

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Загружаем данные для оценки
    # Нам не нужен train_dataloader для скоринга, но функция возвращает все три
    _, eval_dataloader, eval_examples, eval_features = get_squad_dataloaders(
        batch_size=args.batch_size, 
        subset_size=args.subset_size
    )
    
    # 2. Создаем "скелет" модели с правильной структурой (QAT или ADC)
    print(f"Creating model skeleton with structure: '{args.model_structure}'...")
    # Начинаем с базовой FP модели на CPU
    model_skeleton = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
    
    # Адаптируем скелет, чтобы он соответствовал сохраненной модели
    model_skeleton = adapt_model_for_stage(
        model_skeleton,
        stage=args.model_structure, # 'qat' или 'adc'
        bw=args.bw,
        bx=args.bx,
        ba=args.ba,
        k=args.k
    )

    # 3. Загружаем обученные веса в скелет
    print(f"Loading model weights from: {args.model_path}")
    # map_location=device гарантирует, что веса загрузятся сразу на нужное устройство
    model_skeleton.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Перемещаем всю модель на целевое устройство (важно для консистентности)
    model_skeleton.to(device)
    
    # 4. КЛЮЧЕВОЙ ШАГ: Переводим модель в режим оценки.
    # Это отключит наблюдателей и заставит квантизаторы использовать загруженные scale/zero_point.
    model_skeleton.eval()

    # 5. Запускаем оценку
    print("Starting evaluation...")
    metrics = evaluate(model_skeleton, eval_dataloader, eval_examples, eval_features, device)
    
    print("\n----- Evaluation Results -----")
    print(f"Model: {args.model_path}")
    print(f"Structure: {args.model_structure}")
    print(f"Metrics: {metrics}")
    print("----------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a pre-trained BERT Quantized Model")
    
    # Основные аргументы
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model state_dict (.pt file)")
    parser.add_argument("--model_structure", type=str, required=True, choices=['qat', 'adc'], help="The quantization structure of the saved model ('qat' or 'adc')")
    
    # Аргументы для воссоздания структуры модели (должны совпадать с теми, что использовались при обучении)
    parser.add_argument("--bx", type=int, default=4, help="Bit-width for activations used during training")
    parser.add_argument("--bw", type=int, default=4, help="Bit-width for weights used during training")
    parser.add_argument("--ba", type=int, default=8, help="Bit-width for ADC used during training")
    parser.add_argument("--k", type=int, default=4, help="k-factor for ADC used during training")

    # Аргументы для данных
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--subset_size", type=int, default=None, help="Use a subset of the dataset for quick scoring (e.g., 1000)")

    args = parser.parse_args()
    main(args) 