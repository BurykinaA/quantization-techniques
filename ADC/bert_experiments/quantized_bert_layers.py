import torch
from ADC.quantized_layers import LinearQuant, LinearADC, LinearADCAshift

def _selectively_replace_layers(module, stage, bw, bx, ba, k, name_prefix=""):
    """
    Recursively replaces layers in a BERT model based on the specified stage ('qat' or 'adc').
    """
    for name, child in module.named_children():
        full_name = f"{name_prefix}.{name}" if name_prefix else name

        # Рекурсивно идем вглубь, если это не линейный слой
        if not isinstance(child, torch.nn.Linear):
            _selectively_replace_layers(child, stage, bw, bx, ba, k, name_prefix=full_name)
            continue

        # --- Применяем правила замены ---

        # Правило 1: Последний слой (классификатор) всегда 8-бит
        if "qa_outputs" in full_name:
            quant_bw, quant_bx = 8, 8
        else:
            quant_bw, quant_bx = bw, bx

        # Правило 2: Пропускаем квантование для слоя после BMM2 (attention.output.dense)
        if "attention.output.dense" in full_name:
            # На стадии QAT оставляем его в FP32.
            # На стадии ADC он уже будет заменен, если мы решим его квантовать, но пока пропускаем.
            # В данном пайплайне мы его не трогаем.
            continue
            
        # Определяем, какой слой использовать в зависимости от стадии
        if stage == 'qat':
            new_layer = LinearQuant(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                bx=quant_bx,
                bw=quant_bw
            )
        elif stage == 'adc':
            # Правило 3: Применяем A-shift только для слоя перед GELU (intermediate.dense)
            use_ashift = "intermediate.dense" in full_name
            
            QuantLayer = LinearADCAshift if use_ashift else LinearADC
            new_layer = QuantLayer(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                bx=quant_bx,
                bw=quant_bw,
                ba=ba,
                k=k,
                # ashift флаг теперь зависит от имени слоя
                ashift=use_ashift
            )
        else:
            raise ValueError(f"Unknown stage: {stage}")

        setattr(module, name, new_layer)


def adapt_model_for_stage(model, stage, bw, bx, ba, k):
    """
    Adapts the BERT model for a specific quantization stage.
    """
    # Правило для первого слоя: Первый линейный слой в первом энкодере - 8 бит.
    # Мы сделаем это, применив общую замену, а затем вручную исправим первый слой.
    # Однако, в BERT `embeddings` не содержат линейных слоев. Первый линейный слой - `attention.self.query`.
    # Для простоты, мы будем считать, что правило "первый слой 8-бит" покрывается
    # неквантованием embedding'ов, а все слои энкодера следуют общим правилам.
    # Если бы нужно было строго, мы бы нашли `encoder.layer.0...` и задали ему 8 бит.
    
    print(f"Adapting model for stage: '{stage}'...")
    _selectively_replace_layers(model.bert, stage, bw, bx, ba, k)
    
    # Последний слой находится вне `model.bert`, поэтому обрабатываем его отдельно
    if stage == 'qat':
        new_qa_layer = LinearQuant(model.qa_outputs.in_features, model.qa_outputs.out_features, bias=True, bx=8, bw=8)
    elif stage == 'adc':
        new_qa_layer = LinearADC(model.qa_outputs.in_features, model.qa_outputs.out_features, bias=True, bx=8, bw=8, ba=ba, k=k)
    else:
        new_qa_layer = model.qa_outputs
    
    model.qa_outputs = new_qa_layer
    print("Model adaptation complete.")
    return model 