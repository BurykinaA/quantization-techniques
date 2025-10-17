# ADC Post-Training Quantization (PTQ) для BERT

## Что это?

**Post-Training Quantization (PTQ)** — это более простой подход чем QAT:
- ❌ НЕ обучаем веса
- ✅ Только калибруем quantizer параметры (**scales** активаций и весов)
- ⚠️ **Delta остается константой** - это hardware параметр!
- ⚡ Быстро: ~10 минут вместо часов
- 🎯 Часто дает хорошие результаты

## Как это работает

### ⚠️ ВАЖНО: Delta - это аппаратная константа!

**Delta определяется физически схемой ADC и НЕ может быть изменен!**

```
Delta = 2M(2^(bx-1) - 1)(2^(bw-1) - 1) / (2^ba × k)  ← ФИКСИРОВАНО!
```

### Что мы калибруем:
- ✅ **Activation scale (s_x)** - масштаб квантизации активаций
- ✅ **Weight scales (s_w)** - масштабы квантизации весов (per-channel)
- ❌ **НЕ delta** - это hardware параметр!

### Шаг 1: Калибровка
1. Прогоняем небольшую часть данных (~100-1000 samples)
2. Собираем статистику активаций и весов в каждом слое
3. Вычисляем **оптимальные scales**, чтобы значения после integer MM оптимально использовали фиксированный ADC диапазон

### Шаг 2: Применение
1. Устанавливаем откалиброванные **scales**
2. Замораживаем их
3. Оцениваем модель (delta остается как аналитическое значение)

### Методы калибровки

| Метод | Описание | Когда использовать |
|-------|----------|-------------------|
| `minmax` | Использует max(abs(values)) для вычисления scales | Простой, но чувствителен к outliers |
| `percentile` | Использует 99.9-й перцентиль для вычисления scales | **Рекомендуется** - игнорирует outliers |
| `mse` | Минимизирует MSE error при подборе scales | Самый точный, но медленнее |

---

## Команды для запуска

### Базовый запуск (рекомендуется)

```bash
python ADC/bert_my/bert_adc_ptq.py \
  --qat_checkpoint_dir outputs_qa_qat_w_reshape/squad_qat_20251010_132531 \
  --output_dir ./outputs_adc_ptq_base \
  --bx 8 --bw 8 --ba 8 --k 4 \
  --signed_activations \
  --calibration_method percentile \
  --num_calibration_batches 100 \
  --calibration_batch_size 8
```

### С A-shift

```bash
python ADC/bert_my/bert_adc_ptq.py \
  --qat_checkpoint_dir outputs_qa_qat_w_reshape/squad_qat_20251010_132531 \
  --output_dir ./outputs_adc_ptq_ashift \
  --bx 8 --bw 8 --ba 8 --k 4 \
  --signed_activations \
  --ashift \
  --calibration_method percentile \
  --num_calibration_batches 100
```

### Быстрая калибровка (для тестирования)

```bash
python ADC/bert_my/bert_adc_ptq.py \
  --qat_checkpoint_dir outputs_qa_qat_w_reshape/squad_qat_20251010_132531 \
  --output_dir ./outputs_adc_ptq_fast \
  --bx 8 --bw 8 --ba 8 --k 4 \
  --signed_activations \
  --calibration_method minmax \
  --num_calibration_batches 10 \
  --calibration_batch_size 16
```

### Максимальная точность (MSE calibration)

```bash
python ADC/bert_my/bert_adc_ptq.py \
  --qat_checkpoint_dir outputs_qa_qat_w_reshape/squad_qat_20251010_132531 \
  --output_dir ./outputs_adc_ptq_mse \
  --bx 8 --bw 8 --ba 8 --k 4 \
  --signed_activations \
  --calibration_method mse \
  --num_calibration_batches 200
```

### 4-bit квантизация

```bash
python ADC/bert_my/bert_adc_ptq.py \
  --qat_checkpoint_dir outputs_qa_qat_w_reshape/squad_qat_20251010_132531 \
  --output_dir ./outputs_adc_ptq_4bit \
  --bx 4 --bw 4 --ba 4 --k 4 \
  --signed_activations \
  --calibration_method percentile \
  --num_calibration_batches 200
```

---

## Параметры

### Основные
- `--qat_checkpoint_dir` - Путь к QAT checkpoint
- `--output_dir` - Куда сохранить откалиброванную модель
- `--bx`, `--bw`, `--ba`, `--k` - ADC параметры квантизации
- `--signed_activations` - Использовать signed activations (рекомендуется для BERT)

### Калибровка
- `--calibration_method` - Метод: `minmax`, `percentile`, `mse`
- `--num_calibration_batches` - Сколько батчей использовать (100-200 хорошо)
- `--calibration_batch_size` - Размер батча для калибровки

### Опциональные
- `--ashift` - Включить activation shifting
- `--mvm_limit` - Лимит для tiling (default: 256)

---

## Output

После запуска создаются файлы:

```
outputs_adc_ptq_base/
├── pytorch_model.bin          # Откалиброванная модель
├── config.json
├── tokenizer_config.json
├── calibration_info.txt       # Детали калибровки
└── eval_metrics.txt           # F1 и EM scores
```

### Пример calibration_info.txt

```
Calibration method: percentile
Calibration batches: 100
ADC config: bx=8, bw=8, ba=8, k=4
Signed activations: True
A-shift: False

Results:
F1: 85.42
EM: 78.31

Calibrated layers: 144

Per-layer delta values:
  bert.encoder.layer.0.attention.output.dense.tiles.0: delta=245.32, absmax=31155.60
  bert.encoder.layer.0.intermediate.dense.tiles.0: delta=312.45, absmax=39681.15
  ...
```

---

## Сравнение: PTQ vs QAT

| Аспект | PTQ | QAT |
|--------|-----|-----|
| **Время** | ~10 минут | ~3 часа |
| **Простота** | Очень просто | Сложнее |
| **Точность** | Обычно 85-88% F1 | Может быть 88-90% F1 |
| **Стабильность** | Высокая | Может не сойтись |
| **Когда использовать** | Первая попытка, baseline | Когда нужна максимальная точность |

---

## Troubleshooting

### Проблема: F1 < 70

**Решение 1:** Увеличьте число calibration batches
```bash
--num_calibration_batches 500
```

**Решение 2:** Попробуйте MSE calibration
```bash
--calibration_method mse
```

**Решение 3:** Проверьте исходный QAT checkpoint - возможно он сам плохой

### Проблема: Калибровка слишком медленная

**Решение:** Уменьшите число батчей
```bash
--num_calibration_batches 50 --calibration_batch_size 16
```

### Проблема: Out of memory

**Решение:** Уменьшите batch size
```bash
--calibration_batch_size 4 --eval_batch_size 8
```

---

## Рекомендации

1. **Всегда начинайте с PTQ** - это быстро и часто работает хорошо
2. **Используйте `--signed_activations`** для BERT
3. **Метод калибровки:** начните с `percentile`, если не работает → `mse`
4. **Число батчей:** 100-200 обычно достаточно
5. **Если PTQ не работает**, тогда пробуйте QAT

---

## Что делать дальше

После успешной PTQ:

### 1. Сравните с baseline
```bash
# Baseline QAT
python ADC/bert_my/bert_qat_integration.py \
  --fp_checkpoint_dir outputs_qa_qat_w_reshape/squad_qat_20251010_132531 \
  --eval_only

# Ваш ADC PTQ
# Уже есть в eval_metrics.txt
```

### 2. Попробуйте разные конфигурации

Экспериментируйте с:
- Разные методы калибровки
- С/без A-shift
- Разные bit widths

### 3. Визуализируйте результаты

```bash
python ADC/bert_my/run_adc_debug.py \
  --model_path ./outputs_adc_ptq_base \
  --output_dir ./adc_debug_ptq \
  --layers layer.0.attention.output.dense layer.11.output.dense
```

---

## Примеры результатов (ожидаемые)

| Config | Method | F1 | EM |
|--------|--------|----|----|
| 8-bit, signed, percentile | PTQ | ~86 | ~79 |
| 8-bit, signed, mse | PTQ | ~87 | ~80 |
| 8-bit, signed + ashift | PTQ | ~85 | ~78 |
| 4-bit, signed | PTQ | ~82 | ~75 |

*Это приблизительные значения для BERT-base на SQuAD v1.1*

---

## TLDR - Быстрый старт

**Одна команда для начала:**

```bash
python ADC/bert_my/bert_adc_ptq.py \
  --qat_checkpoint_dir outputs_qa_qat_w_reshape/squad_qat_20251010_132531 \
  --output_dir ./outputs_adc_ptq \
  --bx 8 --bw 8 --ba 8 --k 4 \
  --signed_activations \
  --calibration_method percentile \
  --num_calibration_batches 100
```

**Ожидайте:** F1 ~85-87, время ~10 минут ⚡

