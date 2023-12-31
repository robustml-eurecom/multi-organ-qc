import os
import torch
import yaml
import argparse
import numpy as np

from models.CAE.cae import ConvAutoencoder
from models.CAE.small_cae import SmallConvAutoencoder
from models.utils import load_opt_params

from utils.testing import testing
from utils.dataset import DataLoader, NiftiDataset
from utils.preprocess import transform_aug


def main(args):
    DATA_PATH = os.path.join(args.data, args.organ)
    print("+-------------------------------------+")
    print(f'Running in the following path: {DATA_PATH}.')
    print("+-------------------------------------+")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    optimal_parameters = load_opt_params(os.path.join(DATA_PATH, "preprocessed"), model=args.model.lower())
    
    transform, _ = transform_aug(num_classes=optimal_parameters["in_channels"], model=args.model.lower(), size=optimal_parameters["size"][args.organ])
    dataset = NiftiDataset(DATA_PATH+'/structured', transform=transform, mode='test')
    
    model = ConvAutoencoder(keys=config["run_params"]["keys"], 
                         **optimal_parameters
                         ).to(device)
    model.load_checkpoint(data_path=DATA_PATH, eval=True)
    eval_ids = np.load(os.path.join(DATA_PATH,'saved_ids.npy'), allow_pickle=True).item().get('test_ids')
    
    if args.model.lower() == 'cae': model = ConvAutoencoder(keys=config["run_params"]["keys"], 
                            **optimal_parameters
                            ).to(device)
    elif args.model.lower() == 'small_cae': model = SmallConvAutoencoder(keys=config["run_params"]["keys"], 
                            **optimal_parameters
                            ).to(device)
    _ = testing(
        ae=model, 
        data_path=DATA_PATH,
        test_loader=torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8),
        folder_out=os.path.join(DATA_PATH, args.output),
        ids=eval_ids,
        compute_results=False)
    

if __name__ == '__main__':
    # Add command-line arguments
    parser = argparse.ArgumentParser(description='Testing script for MOQC.')
    
    parser.add_argument('-d', '--data', type=str, 
                        default='data', help='Data folder.')
    parser.add_argument('-cf', '--config_file', type=str, 
                        default='moqc/models/config.yml', help='Configuration file.')
    parser.add_argument('-o', '--output', type=str,
                        default='reconstructions', help='Output folder.')
    parser.add_argument('-og', '--organ', type=str, help='Selected organ.')
    parser.add_argument('-m', '--model', type=str, help='Model to be used.')
    parser.add_argument('--verbose', action='store_false', help='Enable verbose mode.')

    args = parser.parse_args()
    
    main(args)



