import argparse
import torch
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import os
import time
import json
import matplotlib.pyplot as plt

from ADC.bert_experiments.bert_experiment_setup import get_model, get_squad_dataloaders
from ADC.bert_experiments.bert_evaluate import evaluate

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_dataloader, eval_dataloader, eval_examples, eval_features = get_squad_dataloaders(args.batch_size, args.subset_size)
    
    # Model
    model = get_model(args.model_type, args.bx, args.bw, args.ba, args.k, args.ashift)
    model.to(device)

    # Lists to store metrics over time
    train_losses = []
    eval_metrics_history = []

    # Training loop
    if args.model_type == 'baseline':
        print("Baseline model selected. Skipping training and proceeding to evaluation.")
    else:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        num_training_steps = args.num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        progress_bar = tqdm(range(num_training_steps))
        for epoch in range(args.num_epochs):
            model.train()
            epoch_loss = 0
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
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            train_losses.append(avg_epoch_loss)
            
            print(f"\nEvaluating at end of epoch {epoch+1}...")
            metrics = evaluate(model, eval_dataloader, eval_examples, eval_features, device)
            eval_metrics_history.append(metrics)
            print(f"Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f}, Eval Metrics: {metrics}")

        print("Training finished.")

    # Evaluation
    print("Starting final evaluation...")
    final_metrics = evaluate(model, eval_dataloader, eval_examples, eval_features, device)
    print(f"Final evaluation results: {final_metrics}")

    # Save model
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_name_parts = [args.model_type]
        if args.model_type != 'baseline':
            model_name_parts.append(f"bx{args.bx}")
            model_name_parts.append(f"bw{args.bw}")
            if args.ashift:
                model_name_parts.append("ashift")
        
        model_name = "_".join(model_name_parts)
        output_dir = os.path.join("bert_models", f"{model_name}_{timestamp}")
        print(f"No --output_dir specified. Saving model to default directory: {output_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    print(f"Model saved to {output_dir}")

    # Save metrics and plots
    results_payload = {
        'args': vars(args),
        'final_metrics': final_metrics
    }
    
    if args.model_type != 'baseline':
        results_payload['train_losses'] = train_losses
        results_payload['eval_metrics_history'] = eval_metrics_history

        # Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, args.num_epochs + 1), train_losses, marker='o')
        plt.title(f'Training Loss per Epoch - {args.model_type}')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "training_loss.png"))
        plt.close()
        
        # Plot evaluation metrics
        f1_scores = [m['f1'] for m in eval_metrics_history]
        exact_matches = [m['exact_match'] for m in eval_metrics_history]
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, args.num_epochs + 1), f1_scores, marker='o', label='F1 Score')
        plt.plot(range(1, args.num_epochs + 1), exact_matches, marker='s', label='Exact Match')
        plt.title(f'Evaluation Metrics per Epoch - {args.model_type}')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "evaluation_metrics.png"))
        plt.close()
        
        print(f"Metrics plots saved to {output_dir}")

    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(results_payload, f, indent=4)
    
    print(f"Results and configuration saved to {os.path.join(output_dir, 'results.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT SQUAD Experiment")
    parser.add_argument("--model_type", type=str, default="baseline", choices=["baseline", "adc", "adc_ashift"], help="Model type")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save model and results")
    
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
    parser.add_argument("--subset_size", type=int, default=None, help="Use a subset of the dataset for quick testing")

    args = parser.parse_args()
    main(args) 