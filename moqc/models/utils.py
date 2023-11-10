import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from torch.autograd import Variable

from medpy.metric import binary
from .CAE.cae import ConvAutoencoder

#use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##################
# Plot utilities #
##################

def plot_history(history):
    if not isinstance(history, (dict, list)):
        raise ValueError("Input history should be a dictionary or a list")
    
    if isinstance(history, dict):
        history = [[x['Total'] for x in history]]
    
    num_losses = len(history)
    # Create subplots
    fig, axes = plt.subplots(1, num_losses, figsize=(15, 5))
    
    if num_losses == 1: axes = [axes]  # Ensure that axes is a list, even for a single subplot
    
    for i, loss in enumerate(history):
        label = f'Loss {i + 1}' if num_losses > 1 else 'Loss'
        axes[i].plot(loss, label=label)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Loss')
        axes[i].legend()
        axes[i].set_title(f'{label} vs. No. of epochs')
        axes[i].grid()

    plt.tight_layout()  # To prevent overlapping subplots
    plt.savefig('logs/train_loss.png')
    plt.show()


def htlm_images(img_list, output_path):
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    html_content = HTML(ani.to_jshtml())

    with open(output_path, 'w') as f:
        f.write(html_content.data)


#######################
#Hyperparameter Tuning#
#######################

def get_sets(parameters, set_parameters=None):
    if set_parameters is None:
        set_parameters = {k: None for k in parameters.keys()}
    if None not in set_parameters.values():
        yield set_parameters
    else:
        current_index = list(set_parameters.values()).index(None)
        current_parameter = list(set_parameters.keys())[current_index]
        for value in parameters[current_parameter]:
            set_parameters[current_parameter] = value
            loader = get_sets(parameters, set_parameters=set_parameters.copy())
            while True:
                try:
                    yield next(loader)
                except StopIteration:
                    break 

def satisfies_rules(rules, set_parameters):
    for rule in rules:
        keys = np.unique(rule.split('"')[1::2])
        for key in keys:
            if key in set_parameters:
                rule = rule.replace('"' + key + '"', 'set_parameters["' + key + '"]')
        if not eval(rule):
            return False
    return True


# TOCHANGE: is harcoded for heart key values
def hyperparameter_tuning(parameters, train_loader, val_loader, transform, transform_augmentation, rules=[], fast=False):
    best_dc = 0
    optimal_parameters = None
    for set_parameters in get_sets(parameters):
        if not satisfies_rules(rules, set_parameters):
            continue
        print(set_parameters)

        BATCH_SIZE = set_parameters["BATCH_SIZE"]
        DA = set_parameters["DA"]
        train_loader.set_batch_size(BATCH_SIZE)
        val_loader.set_batch_size(BATCH_SIZE)
        train_loader.set_transform(transform_augmentation if DA else transform)
        val_loader.set_transform(transform)

        ae = ConvAutoencoder(**set_parameters).to(device)
        
        history = ae.training_routine(
            range(0,set_parameters["tuning_epochs"]),
            train_loader,
            val_loader
        )
        history = {k:[x[k] for x in history] for k in history[0].keys() if k in ["LV_dc", "MYO_dc", "RV_dc"]}
        history = pd.DataFrame.from_dict(history)
        
        wasBlack = any(np.all(history.values==0, axis=1))
        isNotBlack = all(history.values[-1] > 0.01)
        avg_dc = np.mean(history.values[-1])

        if wasBlack and isNotBlack:
            if avg_dc > best_dc:
                best_dc = avg_dc
                optimal_parameters = set_parameters.copy()
            if fast:
                break

    return optimal_parameters

def load_opt_params(prepro_path: str, model:str):
    optimal_parameters = np.load(
        os.path.join(
            prepro_path, 
            f"{model}_optimal_parameters.npy"), 
        allow_pickle=True).item()

    assert optimal_parameters is not None, "Be sure to continue with a working set of hyperparameters"
    
    return optimal_parameters


def reparameterization(latent_dim, mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(torch.FloatTensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    z = sampled_z * std + mu
    return z