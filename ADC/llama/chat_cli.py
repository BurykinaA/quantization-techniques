"""
ADC Model CLI Chat

Usage:
    python ADC/llama/chat_cli.py --checkpoints-dir <path>
"""

import sys
import json
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def discover_models(checkpoints_dir):
    models = {}
    for d in sorted(Path(checkpoints_dir).iterdir()):
        if not (d / "model_full.pt").exists() or not (d / "model_info.json").exists():
            continue
        info = json.loads((d / "model_info.json").read_text())
        info["pt_path"] = str(d / "model_full.pt")
        info["checkpoint_dir"] = str(d)
        models[d.name] = info
    return models


def pick_model(models):
    names = list(models.keys())
    print("\nAvailable models:")
    for i, name in enumerate(names):
        info = models[name]
        r = info.get("results", {})
        wiki = r.get("ppl_adc_wikitext2")
        c4   = r.get("ppl_adc_c4")
        lora = f"LoRA r{info['lora_rank']}" if info.get("lora_rank", 0) > 0 else "no LoRA"
        wiki_str = f"{wiki:.2f}" if wiki else "—"
        c4_str   = f"{c4:.2f}"   if c4   else "—"
        print(f"  [{i+1}] {name}")
        print(f"       mvm={info.get('mvm_limit')}  {lora}  wiki_adc={wiki_str}  c4_adc={c4_str}")
    print()
    while True:
        try:
            choice = int(input("Select model number: ").strip())
            if 1 <= choice <= len(names):
                return names[choice - 1], models[names[choice - 1]]
        except (ValueError, KeyboardInterrupt):
            pass
        print(f"Enter a number between 1 and {len(names)}.")


def load_model(info):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {info['pt_path']} onto {device} ...")
    model = torch.load(info["pt_path"], map_location="cpu", weights_only=False)
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(info["checkpoint_dir"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Ready.\n")
    return model, tokenizer, device


def generate(model, tokenizer, device, history, user_message):
    parts = []
    for u, a in history:
        parts.append(f"User: {u}")
        parts.append(f"Assistant: {a}")
    parts.append(f"User: {user_message}")
    parts.append("Assistant:")
    prompt = "\n".join(parts)

    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", required=True)
    args = parser.parse_args()

    models = discover_models(args.checkpoints_dir)
    if not models:
        print("No models found.")
        sys.exit(1)

    model_name, info = pick_model(models)
    model, tokenizer, device = load_model(info)

    print(f"Chatting with: {model_name}")
    print("Type 'quit' to exit, 'clear' to reset history.\n")

    history = []
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() == "quit":
            break
        if user.lower() == "clear":
            history.clear()
            print("History cleared.\n")
            continue

        response = generate(model, tokenizer, device, history, user)
        print(f"Model: {response}\n")
        history.append((user, response))
        if len(history) > 20:
            history = history[-20:]


if __name__ == "__main__":
    main()
