"""
ADC Model Chat Server (Gradio)

Usage:
    python ADC/llama/serve_chat.py --checkpoints-dir ADC/llama/checkpoints/test_YYYYMMDD

Each model runs in its full ADC quantization setup (TiledLinearADC forward path).
Models are loaded lazily — one at a time, previous is unloaded from GPU.
"""

import sys
import json
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Make custom ADC/LoRA modules importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gradio as gr


# ─── Model registry + lazy loader ────────────────────────────────────────────

_registry: dict = {}       # display_name → info dict
_model        = None
_tokenizer    = None
_loaded_name  = ""
_device       = "cuda" if torch.cuda.is_available() else "cpu"


def discover_models(checkpoints_dir: str) -> dict:
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
        print(f"  {name}  |  mvm={info.get('mvm_limit')}  "
              f"wiki_adc={r.get('ppl_adc_wikitext2','?'):.2f}  "
              f"c4_adc={r.get('ppl_adc_c4','?'):.2f}")
    return models


def load_model(display_name: str):
    global _model, _tokenizer, _loaded_name
    if display_name == _loaded_name and _model is not None:
        return
    print(f"[loader] unloading '{_loaded_name}' ...")
    if _model is not None:
        del _model
        _model = None
        _tokenizer = None
        _loaded_name = ""
        if _device == "cuda":
            torch.cuda.empty_cache()

    info = _registry[display_name]
    print(f"[loader] loading '{display_name}' from {info['pt_path']} ...")
    _model = torch.load(info["pt_path"], map_location="cpu", weights_only=False)
    _model = _model.to(_device)
    _model.eval()
    _tokenizer = AutoTokenizer.from_pretrained(info["checkpoint_dir"])
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    _loaded_name = display_name
    print(f"[loader] '{display_name}' ready on {_device}")


# ─── Generation ──────────────────────────────────────────────────────────────

def build_prompt(history: list[tuple], user_message: str) -> str:
    parts = []
    for user_turn, bot_turn in history:
        parts.append(f"User: {user_turn}")
        if bot_turn:
            parts.append(f"Assistant: {bot_turn}")
    parts.append(f"User: {user_message}")
    parts.append("Assistant:")
    return "\n".join(parts)


def chat(
    user_message: str,
    history: list[tuple],
    model_name: str,
    temperature: float,
    max_new_tokens: int,
):
    if not model_name:
        yield history + [(user_message, "⚠ Select a model first.")]
        return

    load_model(model_name)

    prompt = build_prompt(history, user_message)
    dev    = next(_model.parameters()).device
    inputs = _tokenizer(prompt, return_tensors="pt",
                        truncation=True, max_length=1024).to(dev)

    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=temperature > 0,
            temperature=float(temperature) if temperature > 0 else 1.0,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=_tokenizer.eos_token_id,
        )

    new_ids = out[0][inputs["input_ids"].shape[1]:]
    response = _tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    yield history + [(user_message, response)]


# ─── Model info panel ────────────────────────────────────────────────────────

def model_info_md(model_name: str) -> str:
    if not model_name or model_name not in _registry:
        return "_No model selected._"
    info = _registry[model_name]
    r    = info.get("results", {})

    def fmt(v):
        return f"{v:.2f}" if v is not None else "—"

    lora = f"r{info['lora_rank']} · {info.get('lora_loss','?')} · {', '.join(info.get('lora_targets', []))}" \
           if info.get("lora_rank", 0) > 0 else "none"

    return (
        f"| | |\n|---|---|\n"
        f"| **mvm_limit** | {info.get('mvm_limit')} |\n"
        f"| **delta (approx)** | {2 * info.get('mvm_limit',256) * 127 * 127 / (256 * 16):.0f} |\n"
        f"| **LoRA** | {lora} |\n"
        f"| **wiki bypass PPL** | {fmt(r.get('ppl_bypass_wikitext2'))} |\n"
        f"| **wiki ADC PPL** | {fmt(r.get('ppl_adc_wikitext2'))} |\n"
        f"| **C4 bypass PPL** | {fmt(r.get('ppl_bypass_c4'))} |\n"
        f"| **C4 ADC PPL** | {fmt(r.get('ppl_adc_c4'))} |\n"
    )


# ─── Gradio UI ───────────────────────────────────────────────────────────────

def build_ui(model_names: list[str]) -> gr.Blocks:
    with gr.Blocks(title="ADC LLaMA Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## ADC Quantized LLaMA Chat\nSelect a model, then chat with it in its full ADC quantization setup.")

        with gr.Row():
            with gr.Column(scale=1):
                model_dd = gr.Dropdown(
                    choices=model_names,
                    label="Model",
                    value=model_names[0] if model_names else None,
                    interactive=True,
                )
                info_md = gr.Markdown(
                    model_info_md(model_names[0]) if model_names else "",
                    label="Model info",
                )
                temperature = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Temperature")
                max_tokens  = gr.Slider(32, 512, value=256, step=32, label="Max new tokens")

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, label="Chat")
                msg_box = gr.Textbox(
                    placeholder="Type a message and press Enter…",
                    label="Message",
                    lines=2,
                )
                with gr.Row():
                    send_btn  = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear")

        # Update info panel when model changes
        model_dd.change(model_info_md, inputs=model_dd, outputs=info_md)

        # Send on button click or Enter
        submit_args = dict(
            fn=chat,
            inputs=[msg_box, chatbot, model_dd, temperature, max_tokens],
            outputs=chatbot,
        )
        send_btn.click(**submit_args).then(lambda: "", outputs=msg_box)
        msg_box.submit(**submit_args).then(lambda: "", outputs=msg_box)

        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_box])

    return demo


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", required=True,
                        help="Dir with model subdirectories (model_full.pt + model_info.json)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link")
    args = parser.parse_args()

    print(f"Discovering models in: {args.checkpoints_dir}")
    _registry.update(discover_models(args.checkpoints_dir))
    if not _registry:
        print("No models found. Run run_test.sh first.")
        sys.exit(1)

    print(f"\n{len(_registry)} model(s) found. Device: {_device}")
    ui = build_ui(list(_registry.keys()))
    ui.launch(server_port=args.port, share=args.share)
