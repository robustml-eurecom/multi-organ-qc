import os
import torch
import yaml

from models.ConvAE.cae import ConvAutoencoder
from models.utils import load_opt_params

from utils.testing import testing
from utils.dataset import DataLoader
from utils.preprocess import transform_aug

ORGAN = 'spleen'
DATA_PATH = os.path.join("data", ORGAN)
CONFIG_FILENAME = "moqc/models/config.yml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    with open(CONFIG_FILENAME, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            
    KEYS = config["run_params"]["keys"]
    
    prepro_path = os.path.join(DATA_PATH, "preprocessed")

    optimal_parameters = load_opt_params(prepro_path)
    transform, _ = transform_aug(num_classes=optimal_parameters['in_channels'])
    
    #ae = AE(keys=KEYS, **optimal_parameters).to(device)
    ae = ConvAutoencoder(keys=KEYS, **optimal_parameters).to(device)
    ae.load_checkpoint(data_path=DATA_PATH, eval=True)
    
    BATCH_SIZE = optimal_parameters['BATCH_SIZE']

    
    _ = testing(
        ae=ae, 
        data_path=DATA_PATH,
        test_loader=DataLoader(data_path=DATA_PATH, mode='test', batch_size=BATCH_SIZE, transform=transform),
        folder_out=os.path.join(DATA_PATH, 'reconstructions'),
        compute_results=False)
    

if __name__ == '__main__':
    main()



