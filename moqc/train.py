import os
import numpy as np
import yaml 
import argparse

import torch
import torch.nn as nn
import lpips

from models.CAE.cae import ConvAutoencoder
from models.CAE.small_cae import SmallConvAutoencoder
from models.GAN.dcgan import DCGAN
from models.utils import plot_history, htlm_images
from models.loss import BKGDLoss, BKMSELoss, SSIMLoss, GDLoss

from utils.preprocess import transform_aug
from utils.dataset import NiftiDataset


def extract_info_for_multitrain(args, data_path, config): 
    with open(os.path.join(data_path, 'preprocessed_classes.yml'), 'r') as file:
        try:
            preprocessed_classes = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    max_class = max([elem for sublist in [d['classes'] for d in preprocessed_classes.values()] for elem in sublist]) + 1
    size = min([config["parameters"][args.model.lower()]["size"][og] for og in args.organ])
    batch_size = max([config["parameters"][args.model.lower()]["batch_size"][og] for og in args.organ])
    return max_class, size, batch_size

def main(args):
    DATA_PATH = os.path.join(args.data, '_'.join(args.organ)) if len(args.organ) > 1 else os.path.join(args.data, args.organ[0])
    print("+-------------------------------------+")
    print(f'Running in the following path: {DATA_PATH}.')
    print("+-------------------------------------+")
    with open(args.config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    parameters = config["parameters"][args.model.lower()]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}.")
    
    assert os.path.exists(os.path.join(DATA_PATH, "saved_ids.npy")), "No saved ids were found. Please run the data_preparation script first."

    prepro_path = os.path.join(DATA_PATH, "preprocessed")
    
    if args.custom_params:
        optimal_parameters = np.load(
            os.path.join(
                prepro_path, f"{args.model.lower()}_optimal_parameters.npy"), 
            allow_pickle=True).item()
    else:
        optimal_parameters = parameters
        optimal_parameters['classes'], optimal_parameters['size'], optimal_parameters['batch_size'] = extract_info_for_multitrain(args, DATA_PATH, config)
        
        if "in_channels" not in optimal_parameters: optimal_parameters["in_channels"] = optimal_parameters["out_channels"] = optimal_parameters["classes"]
        optimal_parameters["functions"] = {
            "BKGDLoss": BKGDLoss(),
            "GDLoss": GDLoss(),
            "MSELoss": nn.MSELoss(),
            }
        optimal_parameters["optimizer"] = torch.optim.Adam
        np.save(os.path.join(prepro_path, f"{args.model.lower()}_optimal_parameters"), optimal_parameters)

    assert optimal_parameters is not None, "Be sure to continue with a working set of hyperparameters"

    transform, _ = transform_aug(size=optimal_parameters["size"], num_classes=optimal_parameters["in_channels"], model=args.model.lower())  

    train_dataset = NiftiDataset(DATA_PATH+'/structured', transform=transform, mode='train')
    val_dataset = NiftiDataset(DATA_PATH+'/structured', transform=transform, mode='val')
    
    if args.model.lower() == "dcgan": 
        model = DCGAN(**optimal_parameters).to(device)
    elif args.model.lower() == "cae": 
        keys = config['run_params']['keys'] 
        if optimal_parameters['classes'] > 2: keys += [f'K{i}' for i in range(2, optimal_parameters['classes'])]
        DA = optimal_parameters["DA"]
        model = ConvAutoencoder(keys=keys, **optimal_parameters).to(device)
    elif args.model.lower() == "small_cae": 
        keys = config['run_params']['keys'] 
        if optimal_parameters['classes'] > 2: keys += [f'K{i}' for i in range(2, optimal_parameters['classes'])]
        DA = optimal_parameters["DA"]
        model = SmallConvAutoencoder(keys=keys, **optimal_parameters).to(device)
    print(model)
    
    ckpt = None
    if ckpt is not None:
        ckpt = torch.load(ckpt)
        model.load_state_dict(ckpt[args.model.lower()])
        model.optimizer.load_state_dict(ckpt[f"{args.model.lower()}_optim"])
        start = ckpt["epoch"]+1
    else:
        start = 0
          
    history, img_list = model.training_routine(
                range(start, parameters["epochs"]),
                torch.utils.data.DataLoader(train_dataset, batch_size=optimal_parameters['batch_size'], shuffle=True, num_workers=8),
                torch.utils.data.DataLoader(val_dataset, batch_size=optimal_parameters['batch_size'], shuffle=False, num_workers=8),
                ckpt_folder=os.path.join(DATA_PATH, f"checkpoints/{args.model.lower()}"),
            )
    
    plot_history(history)
    htlm_images(img_list, "logs/dcgan.html")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing script for MOQC.')

    # Add command-line arguments
    parser.add_argument('-d', '--data', type=str, 
                        default='data', help='Data folder.')
    parser.add_argument('-cf', '--config_file', type=str, 
                        default='moqc/models/config.yml', help='Configuration file.')
    parser.add_argument('--custom_params', type=bool, default=False, help='Enable custom parameters.')
    parser.add_argument('-og', '--organ', metavar='S', type=str, 
                        nargs='+',help='a list of organs')
    parser.add_argument('-m', '--model', type=str, help='Model to be used.')
    parser.add_argument('--verbose', action='store_false', help='Enable verbose mode.')

    args = parser.parse_args()
    
    main(args)