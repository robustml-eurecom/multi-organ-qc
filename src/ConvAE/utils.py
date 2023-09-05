import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings

from medpy.metric import binary
from ConvAE.model import AE

#use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_history(history):
    losses = [x['Total'] for x in history]
    plt.plot(losses, '-x', label="loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Losses vs. No. of epochs')
    plt.grid()
    plt.show()
    plt.savefig('src/logs/')

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

        ae = AE(**set_parameters).to(device)
        
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

def load_opt_params(prepro_path: str):
    optimal_parameters = np.load(
        os.path.join(
            prepro_path, 
            "optimal_parameters.npy"), 
        allow_pickle=True).item()

    assert optimal_parameters is not None, "Be sure to continue with a working set of hyperparameters"
    
    return optimal_parameters