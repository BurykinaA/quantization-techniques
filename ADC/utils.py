from matplotlib import pyplot as plt
import numpy as np
import torch

def draw_layer_stats(stats, layer_name, batch_size=1):
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    
    data = stats[layer_name]

    #names = [["w", "x", "y_for_adc", "out_gth"], ["wq", "xq", "yq_adc", "out"]]
    names = [["w", "x", "y_for_adc", "out_gth"], ["wq", "xq", 'yq_adc', "out"]]
    delta = data['delta'][0]
    delta2 = delta * (2 ** 7 - 1)
    for i in range(2):
        for j in range(4):
            #print(names[i][j])
            samples = data[names[i][j]][0]
            if names[i][j][0] != 'w':
                samples = samples[:batch_size]
            samples = samples.flatten()
            height = ax[i][j].hist(samples, bins=50)[0].max()
            ax[i][j].set_title(names[i][j] + (" (for one block)" if names[i][j][0] == 'y' else ""))
            if (names[i][j] == 'y_for_adc'):
                ax[i][j].plot([-delta, -delta], [0, height], color='r', linestyle='--')
                ax[i][j].plot([delta, delta], [0, height], color='r', linestyle='--')
                ax[i][j].plot([-delta2, -delta2], [0, height], color='g', linestyle='--')
                ax[i][j].plot([delta2, delta2], [0, height], color='g', linestyle='--')
    fig.suptitle(layer_name)

def draw_layer_stats_y(stats, layer_name, batch_size=1):
    
    data = stats[layer_name]

    num_blocks = len(data['y_for_adc'])
    fig, ax = plt.subplots(2, max(2, num_blocks), figsize=(5 * max(2, num_blocks), 10))

    names = ['y_for_adc', 'yq_adc']
    for i in range(2):
        for j in range(num_blocks):
            #print(names[i][j])
            samples = data[names[i]][j][:batch_size]
            samples = samples.flatten()
            ax[i][j].hist(samples, bins=50)
            ax[i][j].set_title(names[i] + f' block {j + 1}/{num_blocks}')
    fig.suptitle(layer_name)


def draw_layer_stats_qat(stats, layer_name, batch_size=1):
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    
    data = stats[layer_name]

    names = [["w", "x", "", "out_gth"], ["wq", "xq", '', "out"]]
    for i in range(2):
        for j in range(4):
            if (j == 2):
                continue
            samples = data[names[i][j]][0]
            if names[i][j][0] != 'w':
                samples = samples[:batch_size]
            samples = samples.flatten()
            height = ax[i][j].hist(samples, bins=50)[0].max()
            ax[i][j].set_title(names[i][j] + (" (for one block)" if names[i][j][0] == 'y' else ""))
    fig.suptitle(layer_name)

def get_stats(model):
    model.logger.enabled = True
    batch, labels = next(iter(test_loader))
    batch = batch.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        res = model(batch)
    model.logger.enabled = False
    return model.logger.get_stats()

def disable_adc(model):
    for nm, m in model.named_modules():
        #print(nm, print(type(m)))
        # or isinstance(m, BlockedConv2dADC)
        if isinstance(m, BlockedConv2dADC) or isinstance(m, LinearADC):
            #print("here")
            m.disable_adc()
def enable_adc(model):
    for nm, m in model.named_modules():
        if (nm == 'conv1' or nm == 'fc'):
            continue
        #print(nm, print(type(m)))
        # or isinstance(m, BlockedConv2dADC)
        if isinstance(m, BlockedConv2dADC) or isinstance(m, LinearADC):
            #print("here")
            m.enable_adc()