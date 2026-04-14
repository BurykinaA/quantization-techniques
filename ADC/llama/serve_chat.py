"""
ADC Model Chat Server (Gradio)

Usage:
    python ADC/llama/serve_chat.py --checkpoints-dir <path>
"""

import sys
import json
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gradio as gr

# ─── State ───────────────────────────────────────────────────────────────────

_registry = {}
_model = None
_tokenizer = None
_loaded_name = ""
_device = "cuda" if torch.cuda.is_available() else "cpu"


def discover_models(checkpoints_dir):
    models = {}
    for d in sorted(Path(checkpoints_dir).iterdir()):
        info_path = d / "model_info.json"
        pt_path   = d / "model_full.pt"
        if not (d.is_dir() and info_path.exists() and pt_path.exists()):
            continue
        info = json.loads(info_path.read_text())
        info["checkpoint_dir"] = str(d)
        info["pt_path"]        = str(pt_path)
        name = info.get("display_name") or d.name
        models[name] = info
        r = info.get("results", {})
        print(f"  {name}  mvm={info.get('mvm_limit')}  "
              f"wiki_adc={r.get('ppl_adc_wikitext2', '?')}  "
              f"c4_adc={r.get('ppl_adc_c4', '?')}")
    return models


def load_model(name):
    global _model, _tokenizer, _loaded_name
    if name == _loaded_name and _model is not None:
        return
    if _model is not None:
        del _model
        _model = None
        if _device == "cuda":
            torch.cuda.empty_cache()
    info = _registry[name]
    print(f"Loading {name} ...")
    _model = torch.load(info["pt_path"], map_location="cpu", weights_only=False)
    _model = _model.to(_device)
    _model.eval()
    _tokenizer = AutoTokenizer.from_pretrained(info["checkpoint_dir"])
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    _loaded_name = name
    print(f"Ready: {name} on {_device}")


# ─── Generation ──────────────────────────────────────────────────────────────

def chat(message, history, model_name):
    if not model_name:
        yield "Select a model first."
        return
    load_model(model_name)

    prompt_parts = []
    for user_turn, bot_turn in history:
        prompt_parts.append(f"User: {user_turn}")
        if bot_turn:
            prompt_parts.append(f"Assistant: {bot_turn}")
    prompt_parts.append(f"User: {message}")
    prompt_parts.append("Assistant:")
    prompt = "\n".join(prompt_parts)

    inputs = _tokenizer(prompt, return_tensors="pt",
                        truncation=True, max_length=1024).to(_device)
    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=_tokenizer.eos_token_id,
        )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    yield _tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ─── UI ──────────────────────────────────────────────────────────────────────

def build_ui(model_names):
    with gr.Blocks(title="ADC LLaMA Chat") as demo:

        gr.Markdown("# ADC Quantized LLaMA Chat")

        model_dd = gr.Dropdown(
            choices=model_names,
            value=model_names[0] if model_names else None,
            label="Model",
        )

        gr.ChatInterface(
            fn=chat,
            additional_inputs=[model_dd],
            chatbot=gr.Chatbot(height=480),
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print(f"Scanning {args.checkpoints_dir} ...")
    _registry.update(discover_models(args.checkpoints_dir))
    if not _registry:
        print("No models found. Run run_test.sh first.")
        sys.exit(1)

    build_ui(list(_registry.keys())).launch(
        server_port=args.port, share=args.share,
        theme=gr.themes.Default(),
    )
