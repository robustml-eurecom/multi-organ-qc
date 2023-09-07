import sys, importlib
import os
import numpy as np
#importlib.reload(sys.modules['ConvAE'])
#importlib.reload(sys.modules['utils'])

import torch
import torchvision

from ConvAE.model import AE
from ConvAE.config import KEYS
from ConvAE.utils import plot_history

from utils.preprocess import transform_aug
from utils.dataset import DataLoader, train_val_test

DATA_PATH = 'data/liver/'
CUSTOM_PARAMS = False

'''
List of args to be implemented:
    - DATA_PATH/ str
    - CUSTOM_PARAMS / bool
    - augmentation / bool ?? 
'''

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #_,_,_ = train_val_test(data_path=DATA_PATH)

    prepro_path = os.path.join(DATA_PATH, "preprocessed")
    
    if CUSTOM_PARAMS:
        optimal_parameters = np.load(
            os.path.join(
                prepro_path, "optimal_parameters.npy"), 
            allow_pickle=True).item()
    else:
        optimal_parameters = {
            "BATCH_SIZE": 8,
            "DA": False,
            "latent_size": 100,
            "optimizer": torch.optim.Adam,
            "lr": 2e-4,
            "weight_decay": 1e-5,
            "functions": ["BKGDLoss", "BKMSELoss"],
            "settling_epochs_BKGDLoss": 10,
            "settling_epochs_BKMSELoss": 0
        }
        np.save(os.path.join(prepro_path, "optimal_parameters"), optimal_parameters)

    assert optimal_parameters is not None, "Be sure to continue with a working set of hyperparameters"

    BATCH_SIZE = optimal_parameters["BATCH_SIZE"]
    DA = optimal_parameters["DA"]

    #train_ids = np.load('./data/liver/saved_ids.npy', allow_pickle=True).item().get('train_ids')
    #val_ids = np.load('./data/liver/saved_ids.npy', allow_pickle=True).item().get('val_ids')
    
    ae = AE(keys=KEYS, **optimal_parameters).to(device)

    ckpt = None
    if ckpt is not None:
        ckpt = torch.load(ckpt)
        ae.load_state_dict(ckpt["AE"])
        ae.optimizer.load_state_dict(ckpt["AE_optim"])
        start = ckpt["epoch"]+1
    else:
        start = 0
    
    transform, transform_augmentation = transform_aug()
    
    plot_history(
        ae.training_routine(
            range(start,500),
            DataLoader(data_path=DATA_PATH, mode='train', batch_size=BATCH_SIZE, transform=transform_augmentation if DA else transform),
            DataLoader(data_path=DATA_PATH, mode='test', batch_size=BATCH_SIZE, transform=transform),
            os.path.join(DATA_PATH, "checkpoints/")
        )
    )
    
if __name__ == '__main__':
    main()