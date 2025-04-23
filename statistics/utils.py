import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def load_activations(layer_name, stats_dir='activation_stats/combined'):
    path = os.path.join(stats_dir, f'{layer_name.replace(".", "_")}.npy')
    return np.load(path)

def get_all_layer_names(stats_dir='activation_stats/combined'):
    """Возвращает список всех имён слоёв по файлам в stats_dir."""
    return [os.path.splitext(f)[0].replace('_', '.')
            for f in os.listdir(stats_dir) if f.endswith('.npy')]


import numpy as np
import plotly.graph_objects as go
import os

def plot_histogram(
    layer_name, 
    stats_dir='activation_stats/combined', 
    nbins=100, 
    percentile=None
):
    """
    Гистограмма значений:
      - percentile=None — одноцветная
      - percentile=(p_low, p_high) — всё вне [p_low, p_high] красится красным
    Бины задаются одинаково в обоих режимах: от min(data) до max(data) ровно nbins штук.
    """
    # Загрузка данных
    path = os.path.join(stats_dir, f'{layer_name.replace(".", "_")}.npy')
    acts = np.load(path)
    data = acts.flatten() if acts.ndim == 2 else acts

    # Вычисляем границы и размер бина
    dmin, dmax = data.min(), data.max()
    bin_size = (dmax - dmin) / nbins
    xbins = dict(start=dmin, end=dmax, size=bin_size)

    fig = go.Figure()

    if percentile:
        p_low, p_high = np.percentile(data, percentile)
        mask_mid  = (data >= p_low) & (data <= p_high)
        data_mid  = data[mask_mid]
        data_tail = data[~mask_mid]
        # Центр
        fig.add_trace(go.Histogram(
            x=data_mid, xbins=xbins,
            name=f'{percentile[0]}–{percentile[1]}%', 
            marker_color='blue', opacity=0.75
        ))
        # Хвосты
        fig.add_trace(go.Histogram(
            x=data_tail, xbins=xbins,
            name=f'outside {percentile[0]}–{percentile[1]}%', 
            marker_color='red', opacity=0.75
        ))
        fig.update_layout(barmode='overlay')
    else:
        # Один набор — просто все данные
        fig.add_trace(go.Histogram(
            x=data, xbins=xbins,
            marker_color='blue', opacity=0.75,
            name=layer_name
        ))

    fig.update_layout(
        title=f'Activation Histogram: {layer_name}',
        xaxis_title='Value',
        yaxis_title='Count'
    )
    fig.show()



def filter_layers_by_type(layer_names, layer_type):
    """
    Фильтрация имён слоёв по типу:
      - 'conv': сверточные
      - 'bn': BatchNorm
      - 'relu': ReLU
      - 'pool': MaxPool
      - 'linear': fully-connected
    """
    mapping = {
        'conv': '.conv',
        'bn': '.bn',
        'relu': '.relu',
        'pool': '.maxpool',
        'linear': 'fc'
    }
    key = mapping.get(layer_type)
    if key is None:
        raise ValueError(f'Unknown layer_type {layer_type}')
    return [n for n in layer_names if key in n]

def plot_channel_boxplot(
    layer_name, 
    stats_dir='activation_stats/combined', 
    max_channels=16
):
    """
    Box-plot по каналам (Conv2d).
    """
    acts = load_activations(layer_name, stats_dir)
    assert acts.ndim == 2, "Boxplot only for Conv layers"
    C = acts.shape[0]
    subset = acts[:min(C, max_channels)]
    # данные в формате: [{'channel':i, 'value':v}, ...]
    data = []
    for i, vals in enumerate(subset):
        data.extend([{'channel': i, 'value': float(v)} for v in vals])
    fig = px.box(
        data, x='channel', y='value', 
        title=f'Channel-wise Boxplot: {layer_name}'
    )
    fig.update_traces(marker_color='blue')
    fig.show()

def plot_channel_stats(
    layer_name, 
    stats_dir='activation_stats/combined'
):
    """
    Mean и Std per channel (Conv2d).
    """
    acts = load_activations(layer_name, stats_dir)
    assert acts.ndim == 2, "Only for Conv layers"
    means = acts.mean(axis=1)
    stds  = acts.std(axis=1)
    channels = np.arange(len(means))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=channels, y=means, mode='lines+markers', name='mean'
    ))
    fig.add_trace(go.Scatter(
        x=channels, y=stds, mode='lines+markers', name='std'
    ))
    fig.update_layout(
        title=f'Mean & Std per Channel: {layer_name}',
        xaxis_title='Channel', yaxis_title='Value'
    )
    fig.show()

def compute_sparsity(
    layer_name, 
    stats_dir='activation_stats/combined', 
    eps=1e-6
):
    """
    Доля почти нулевых элементов:
      - Conv2d: возвращает массив для каждого канала
      - остальные: одно число
    """
    acts = load_activations(layer_name, stats_dir)
    if acts.ndim == 2:
        return np.mean(np.abs(acts) < eps, axis=1)
    else:
        return float(np.mean(np.abs(acts) < eps))

def summarize_layer_types(stats_dir='activation_stats/combined'):
    """
    Возвращает dict типа: {'conv': [..], 'bn': [..], ...}
    """
    all_layers = get_all_layer_names(stats_dir)
    summary = {}
    for t in ['conv','bn','relu','pool','linear']:
        summary[t] = filter_layers_by_type(all_layers, t)
    return summary

# # Пример использования:
# if __name__ == '__main__':
#     stats_dir = 'activation_stats/combined'
#     # Список по типам:
#     types = summarize_layer_types(stats_dir)
#     print("Conv layers:", types['conv'])
#     print("Linear layers:", types['linear'])
    
#     # Гистограмма conv1, раскраска 5–95%
#     plot_histogram('conv1', stats_dir, nbins=80, percentile=(5,95))
    
#     # Boxplot по первым 8 каналам первого слоя
#     plot_channel_boxplot('layer1.0.conv1', stats_dir, max_channels=8)
    
#     # Mean/Std по каналам
#     plot_channel_stats('layer1.0.conv1', stats_dir)
    
#     # Sparsity
#     sp = compute_sparsity('layer1.0.conv1', stats_dir)
#     print("Sparsity per channel:", sp)
