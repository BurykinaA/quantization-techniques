import numpy as np
from utils import (
    get_all_layer_names,
    load_activations,
)


'''

1. **compute_tail_ratio**  
   — оценивает, какой процент значений активаций выходит за 0.1–99.9 % перцентили.  
   - Если этот «tail_ratio» > 1 %, значит есть заметные выбросы → рекомендуем `HistogramObserver`.

2. **compute_channel_variation**  
   — для сверточных слоёв считает отношение максимального σ к минимальному σ по каналам.  
   - Если > 5× (порог можно менять), динамики каналов сильно различаются → `per_channel`.

3. **auto_scheme_for_layer**  
   — комбинирует оба критерия и формирует рекомендацию `observer` и `scope`.

4. **auto_quantization_schemes**  
   — генерирует сводную таблицу для всех ваших слоёв.

где стоит:
- клиппить хвосты и использовать гистограмму,
- квантизировать по-тензорно или по-канально.

'''


def compute_tail_ratio(data, lower_p=0.1, upper_p=99.9):
    """
    Доля элементов вне [lower_p, upper_p] перцентилей.
    """
    lo, hi = np.percentile(data, [lower_p, upper_p])
    n = data.size
    outside = np.sum((data < lo) | (data > hi))
    return outside / n

def compute_channel_variation(acts):
    """
    Для Conv-слоя: отношение max_std к min_std по каналам.
    Для линейного/1D: возвращает 1.0 (нет каналов).
    """
    if acts.ndim != 2:
        return 1.0
    stds = acts.std(axis=1)
    # избегаем деления на ноль
    stds = stds + 1e-9
    return stds.max() / stds.min()

def auto_scheme_for_layer(layer_name, stats_dir='activation_stats/combined'):
    """
    Возвращает словарь с рекомендацией:
      - observer: 'minmax' или 'histogram'
      - scope:    'per_tensor' или 'per_channel'
      - comment:  пояснение
    """
    acts = load_activations(layer_name, stats_dir)
    data = acts.flatten() if acts.ndim == 2 else acts

    # 1) хвосты
    tail = compute_tail_ratio(data, lower_p=0.1, upper_p=99.9)
    # 2) каналовая разница
    chan_var = compute_channel_variation(acts)

    # решение по observer
    if tail > 0.01:
        observer = 'histogram'   # >1% выбросов — лучше отсечь
    else:
        observer = 'minmax'

    # решение по scope
    if acts.ndim == 2 and chan_var > 5:
        scope = 'per_channel'
    else:
        scope = 'per_tensor'

    comment = []
    if observer=='histogram':
        comment.append(f"tail_ratio={tail:.2%} → отсекаем экстремумы")
    else:
        comment.append(f"tail_ratio={tail:.2%} → простая MinMax")
    if scope=='per_channel':
        comment.append(f"chan_var={chan_var:.1f}× → per-channel")
    else:
        comment.append(f"chan_var={chan_var:.1f}× → per-tensor")

    return {
        'layer': layer_name,
        'observer': observer,
        'scope': scope,
        'note': "; ".join(comment)
    }

def auto_quantization_schemes(stats_dir='activation_stats/combined'):
    """
    Проходит по всем слоям и собирает рекомендации.
    """
    layers = get_all_layer_names(stats_dir)
    results = []
    for L in layers:
        res = auto_scheme_for_layer(L, stats_dir)
        results.append(res)
    return results

# Пример использования:
# if __name__ == '__main__':
#     recs = auto_quantization_schemes('activation_stats/combined')
#     for r in recs:
#         print(f"{r['layer']}: {r['observer']}, {r['scope']}  ({r['note']})")
