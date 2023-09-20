import os
import numpy as np
import torch
from pytorch_lightning import Trainer

from models.ConvAE.conv_ae import AE
from models.ConvAE.conv_ae_iter import ConvAutoencoder
from models.config import KEYS
from models.utils import load_opt_params
from models.VAE.beta_vae import BetaVAE
from models.VAE.experiment import VAEXperiment
from models.config import KEYS, models_setup

from utils.testing import testing
from utils.dataset import DataLoader
from utils.preprocess import transform_aug

ORGAN = 'liver'
DATA_PATH = os.path.join("data", ORGAN)
LIGHTNING_PIPELINE = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    prepro_path = os.path.join(DATA_PATH, "preprocessed")
    transform, _ = transform_aug()

    optimal_parameters = load_opt_params(prepro_path)
    ckpt_path = "checkpoints/liver/epoch=75-step=125552.ckpt"
    
    #ae = AE(keys=KEYS, **optimal_parameters).to(device)
    #ae = ConvAutoencoder(keys=KEYS, **optimal_parameters).to(device)
    #ae.load_checkpoint(data_path=DATA_PATH, eval=True)
    ae = BetaVAE(**models_setup["params"]).to(device)
    experiment = VAEXperiment(vae_model=ae, params=models_setup['exp_params'])
    
    BATCH_SIZE = optimal_parameters['BATCH_SIZE']

    if LIGHTNING_PIPELINE:
        runner = Trainer()
        runner.test(
            model=experiment,
            dataloaders=DataLoader(data_path=DATA_PATH, mode='test', batch_size=BATCH_SIZE, transform=transform, num_workers=8),
            ckpt_path=ckpt_path,
        )
    else:
        _ = testing(
            ae=experiment, 
            data_path=DATA_PATH,
            test_loader=DataLoader(data_path=DATA_PATH, mode='test', batch_size=BATCH_SIZE, transform=transform),
            folder_out=os.path.join(DATA_PATH, 'reconstructions'),
            compute_results=False)
    

if __name__ == '__main__':
    main()



