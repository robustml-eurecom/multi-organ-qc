import os
import numpy as np
import yaml 

import torch
import torch.nn as nn

from models.ConvAE.cae import ConvAutoencoder
from models.utils import plot_history
from models.ConvAE.loss import BKGDLoss, BKMSELoss, SSIMLoss

from utils.preprocess import transform_aug
from utils.dataset import DataLoader, train_val_test

organ = 'spleen'
DATA_PATH = os.path.join("data", organ)
CUSTOM_PARAMS = False
CONFIG_FILENAME = "moqc/models/config.yml"
'''
List of args to be implemented:
    - DATA_PATH/ str
    - CUSTOM_PARAMS / bool
    - augmentation / bool ?? 
'''

def main():  
    with open(CONFIG_FILENAME, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    parameters = config["parameters"]
    KEYS = config["run_params"]["keys"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}.")
    
    _,_,_ = train_val_test(data_path=DATA_PATH, split=[.8, .1, .1]) if not os.path.exists(os.path.join(DATA_PATH, "saved_ids.npy")) else (None, None, None)

    prepro_path = os.path.join(DATA_PATH, "preprocessed")
    
    if CUSTOM_PARAMS:
        optimal_parameters = np.load(
            os.path.join(
                prepro_path, "optimal_parameters.npy"), 
            allow_pickle=True).item()
    else:
        optimal_parameters = parameters
        optimal_parameters["functions"] = {
            "BKGDLoss": BKGDLoss(), 
            #"BKMSELoss": BKMSELoss(),
            "MSELoss": nn.MSELoss(),
            "BCE": nn.BCELoss(),
            }
        optimal_parameters["optimizer"] = torch.optim.AdamW
        np.save(os.path.join(prepro_path, "optimal_parameters"), optimal_parameters)

    assert optimal_parameters is not None, "Be sure to continue with a working set of hyperparameters"

    BATCH_SIZE = optimal_parameters["BATCH_SIZE"]
    DA = optimal_parameters["DA"]

    #ae = AE(keys=KEYS, **optimal_parameters).to(device)
    ae = ConvAutoencoder(keys=KEYS, **optimal_parameters).to(device)
    print(ae)
    
    ckpt = None
    if ckpt is not None:
        ckpt = torch.load(ckpt)
        ae.load_state_dict(ckpt["AE"])
        ae.optimizer.load_state_dict(ckpt["AE_optim"])
        start = ckpt["epoch"]+1
    else:
        start = 0
        
    transform, transform_augmentation = transform_aug(num_classes=parameters["in_channels"])
        
    plot_history(   
        ae.training_routine(
            range(start, parameters["epochs"]),
            DataLoader(data_path=DATA_PATH, mode='train', batch_size=BATCH_SIZE, num_workers=8, transform=transform_augmentation if DA else transform),
            DataLoader(data_path=DATA_PATH, mode='test', batch_size=BATCH_SIZE, num_workers=8, transform=transform),
            os.path.join(DATA_PATH, "checkpoints/")
        )
    )
    
if __name__ == '__main__':
    main()