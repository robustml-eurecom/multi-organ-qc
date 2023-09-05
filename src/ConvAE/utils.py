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

###############
#Checkpointing#
###############


def clean_old_checkpoints(ckpt_folder:str, best_keep:int=2, total_keep:int=10):
    """
        Is called after each training routine: will keep the {best_keep} best checkpoints, and the other most recent checkpoints for a total of {total_keep checkpoints}. 
        Others will be deleted.
        Example: keep 070_best, 077_best, and 0_78, 0_78, ... and some others

        Parameters
        ----------
            ckpt_folder: str
                The string path to the checkpoints folder
            best_keep: int (default = 2)
                The number of last best checkoints to keep
            total_keep: int (default = 10)
                The total number of checkoints to keep, best and not best included
    """
    assert(best_keep <= total_keep)
    assert(os.path.isdir(ckpt_folder))
    poor_keep = total_keep - best_keep
    poor_ckpts = sorted([file for file in os.listdir(ckpt_folder) if "_best" not in file])
    best_ckpts = sorted([file for file in os.listdir(ckpt_folder) if "_best" in file])
    if len(poor_ckpts)+len(best_ckpts)>total_keep :
        if(len(best_ckpts)>best_keep):
            for file in best_ckpts[:-best_keep] :
                file_path = os.path.join(ckpt_folder, file)
                os.remove(file_path)
        if(len(poor_ckpts)>poor_keep):
            for file in poor_ckpts[:-poor_keep]:
                file_path = os.path.join(ckpt_folder, file)
                os.remove(file_path)