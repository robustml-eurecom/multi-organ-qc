import os
import numpy as np
import yaml 
import argparse

import torch
import torch.nn as nn
import lpips

from models.ConvAE.cae import ConvAutoencoder
from models.utils import plot_history
from models.ConvAE.loss import BKGDLoss, BKMSELoss, SSIMLoss, GDLoss

from utils.preprocess import transform_aug
from utils.dataset import DataLoader, train_val_test

parser = argparse.ArgumentParser(description='Testing script for MOQC.')

def main():
    # Add command-line arguments
    parser.add_argument('-d', '--data', type=str, 
                        default='data', help='Data folder.')
    parser.add_argument('-cf', '--config_file', type=str, 
                        default='moqc/models/config.yml', help='Configuration file.')
    parser.add_argument('-o', '--output', type=str,
                        default='reconstructions', help='Output folder.')
    parser.add_argument('-og', '--organ', type=str, help='Selected organ.')
    parser.add_argument('--verbose', action='store_false', help='Enable verbose mode.')
    parser.add_argument('--custom_params', action='store_true', help='Enable custom parameters.')

    args = parser.parse_args()

    DATA_PATH = os.path.join(args.data, args.organ)
     
    with open(args.config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    parameters = config["parameters"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}.")
    
    _,_,_ = train_val_test(data_path=DATA_PATH, split=[.8, .1, .1]) if not os.path.exists(os.path.join(DATA_PATH, "saved_ids.npy")) else (None, None, None)

    prepro_path = os.path.join(DATA_PATH, "preprocessed")
    
    if args.custom_params:
        optimal_parameters = np.load(
            os.path.join(
                prepro_path, "optimal_parameters.npy"), 
            allow_pickle=True).item()
    else:
        optimal_parameters = parameters
        optimal_parameters["in_channels"] = optimal_parameters["out_channels"] = optimal_parameters["classes"][args.organ]
        optimal_parameters["functions"] = {
            "BKGDLoss": BKGDLoss(),
            "MSELoss": nn.MSELoss(),
            "GDLoss": GDLoss(),
            #"BKMSELoss": BKMSELoss(),
            #"CELoss": nn.CrossEntropyLoss(),
            #"LPIPS": lpips.LPIPS().cuda()
            }
        optimal_parameters["optimizer"] = torch.optim.Adam
        np.save(os.path.join(prepro_path, "optimal_parameters"), optimal_parameters)

    assert optimal_parameters is not None, "Be sure to continue with a working set of hyperparameters"

    DA = optimal_parameters["DA"]

    #ae = AE(keys=KEYS, **optimal_parameters).to(device)
    ae = ConvAutoencoder(keys=config['run_params']['keys'], **optimal_parameters).to(device)
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
            DataLoader(data_path=DATA_PATH, mode='train', batch_size=parameters['batch_size'], num_workers=8, transform=transform_augmentation if DA else transform),
            DataLoader(data_path=DATA_PATH, mode='test', batch_size=parameters['batch_size'], num_workers=8, transform=transform),
            os.path.join(DATA_PATH, "checkpoints/")
        )
    )
    
if __name__ == '__main__':
    main()