import os
import torch
import yaml
import argparse

from models.ConvAE.cae import ConvAutoencoder
from models.utils import load_opt_params

from utils.testing import testing
from utils.dataset import DataLoader
from utils.preprocess import transform_aug


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

    args = parser.parse_args()

    DATA_PATH = os.path.join(args.data, args.organ)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
                
    prepro_path = os.path.join(DATA_PATH, "preprocessed")
    optimal_parameters = load_opt_params(prepro_path)
    transform, _ = transform_aug(num_classes=optimal_parameters['in_channels'])
    
    #ae = AE(keys=KEYS, **optimal_parameters).to(device)
    ae = ConvAutoencoder(keys=config["run_params"]["keys"], 
                         **optimal_parameters
                         ).to(device)
    ae.load_checkpoint(data_path=DATA_PATH, eval=True)
    
    _ = testing(
        ae=ae, 
        data_path=DATA_PATH,
        test_loader=DataLoader(data_path=DATA_PATH, mode='test', batch_size=optimal_parameters['batch_size'], transform=transform),
        folder_out=os.path.join(DATA_PATH, args.output),
        compute_results=False)
    

if __name__ == '__main__':
    main()



