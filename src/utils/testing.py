"""
Created on Thu Aug  1 14:58:26 2024

@author: metz
"""
import numpy as np
import torch

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def RMSEP(y_true, y_pred):
    loss = np.sqrt(np.mean(np.square((y_true - y_pred)), axis=0))
    return loss


def ccc(y_true, y_pred):
    # Means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    # Population variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    # Population standard deviations
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    cor = np.corrcoef(y_true, y_pred, rowvar=False)[0][1]

    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    ccc = numerator / denominator
    return (ccc)


def test(model, model_path, test_loader, config):
    Y = []
    y_pred = []
    model.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    with torch.no_grad():
        for inputs, targets in test_loader:
            Y += targets.to("cpu")
            inputs = inputs.to(device, non_blocking=True).float()
            outputs = model(inputs[:, None])
            y_pred += outputs.to("cpu")

    Y = np.array(Y)
    y_pred = np.array(y_pred)

    if config.classification:
        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
        from sklearn.metrics import classification_report, confusion_matrix
        y_pred = np.argmax(y_pred, axis=1)
        Y = np.argmax(Y, axis=1)
        print(classification_report(Y, y_pred,zero_division=True))
        print(confusion_matrix(Y, y_pred))

    else:
        for i in range(Y.shape[1]):
            ccc_score = ccc(y_pred[:, i], Y[:, i])
            r2_score_ = r2_score(Y[:, i], y_pred[:, i])
            rmsep_score = RMSEP(y_pred[:, i], Y[:, i])

            print(f"CCC: {ccc_score}, R2: {r2_score_}, RMSEP: {rmsep_score}")

            plt.figure(figsize=(8, 6))

            # Scatter plot of X vs Y
            print(f"Shape of Y: {Y[:, i].shape}, Shape of y_pred: {y_pred[:, i].shape}")
            plt.scatter(Y[:, i], y_pred[:, i], edgecolors='k', alpha=0.5)

            # Plot of the 45 degree line
            plt.plot([Y.min() - 1, Y.max() + 1], [Y.min() - 1, Y.max() + 1], 'r')
            # add text with cc_score and r2_score
            plt.text(0.95, 0.05, f'CCC: {ccc_score:.2f}\nR²: {r2_score_:.2f}',
                     transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
                     color='red', fontweight='bold', fontfamily='serif')

            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel('Observed', fontsize=16)
            plt.ylabel('Predicted', fontsize=16)
            plt.title(f'Predicted vs Observed for y{i}', fontsize=16)

            plt.show(block=False)
            
    
    
def test_benchmark(model, model_path, test_loader, config):
    Y = []
    y_pred = []
    model.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    with torch.no_grad():
        for inputs, targets in test_loader:
            Y += targets.to("cpu")
            inputs = inputs.to(device, non_blocking=True).float()
            outputs = model(inputs[:, None])
            y_pred += outputs.to("cpu")

    Y = np.array(Y)
    y_pred = np.array(y_pred)

    return(Y, y_pred)
