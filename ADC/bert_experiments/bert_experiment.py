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