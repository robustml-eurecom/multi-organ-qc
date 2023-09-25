import os
import torch

from trash.conv_ae import AE
from models.ConvAE.cae import ConvAutoencoder
from models.config import KEYS
from models.utils import load_opt_params
from models.config import KEYS

from utils.testing import testing
from utils.dataset import DataLoader
from utils.preprocess import transform_aug

ORGAN = 'liver'
DATA_PATH = os.path.join("data", ORGAN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    prepro_path = os.path.join(DATA_PATH, "preprocessed")
    transform, _ = transform_aug()

    optimal_parameters = load_opt_params(prepro_path)
    
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


