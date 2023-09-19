import sys, importlib
import os
import numpy as np

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from models.ConvAE.conv_ae import AE
from models.ConvAE.conv_ae_iter import ConvAutoencoder
from models.VAE.beta_vae import BetaVAE
from models.VAE.experiment import VAEXperiment
from models.config import KEYS, models_setup
from models.utils import plot_history
from models.ConvAE.loss import BKGDLoss, BKMSELoss, SSIMLoss

from utils.preprocess import transform_aug
from utils.dataset import DataLoader, train_val_test

organ = 'liver'
DATA_PATH = os.path.join("data", organ)
CUSTOM_PARAMS = False
LIGHTNING_PIPELINE = True

'''
List of args to be implemented:
    - DATA_PATH/ str
    - CUSTOM_PARAMS / bool
    - augmentation / bool ?? 
'''

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}.")
    _,_,_ = train_val_test(data_path=DATA_PATH) if not os.path.exists(os.path.join(DATA_PATH, "saved_ids.npy")) else (None, None, None)

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
            "in_channels": 4,
            "out_channels": 4,
            "latent_channels": 100,
            "activation": "leaky_relu",
            "optimizer": torch.optim.Adam,
            "lr": 2e-4,
            "weight_decay": 1e-5,
            "functions": {
                "BKGDLoss": BKGDLoss(), 
                "BKMSELoss": BKMSELoss(),
                "SSIM": SSIMLoss()
                },
            "settling_epochs_BKGDLoss": 10,
            "settling_epochs_BKMSELoss": 0
        }
        np.save(os.path.join(prepro_path, "optimal_parameters"), optimal_parameters)

    assert optimal_parameters is not None, "Be sure to continue with a working set of hyperparameters"

    BATCH_SIZE = optimal_parameters["BATCH_SIZE"]
    DA = optimal_parameters["DA"]

    #ae = AE(keys=KEYS, **optimal_parameters).to(device)
    #ae = ConvAutoencoder(keys=KEYS, **optimal_parameters).to(device)
    ae = BetaVAE(**models_setup["params"]).to(device)
    experiment = VAEXperiment(ae, models_setup['exp_params'])
    print(ae)
    
    ckpt = None
    if ckpt is not None:
        ckpt = torch.load(ckpt)
        ae.load_state_dict(ckpt["AE"])
        ae.optimizer.load_state_dict(ckpt["AE_optim"])
        start = ckpt["epoch"]+1
    else:
        start = 0
    transform, transform_augmentation = transform_aug()
    
    if LIGHTNING_PIPELINE:
        tb_logger =  TensorBoardLogger(
            save_dir=models_setup['logging_params']['save_dir'],
            name=models_setup['logging_params']['name']
            )
        
        runner = Trainer(
                logger=tb_logger,
                max_epochs=500,
                callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join("checkpoints", "ogran"), 
                                     monitor= "val_loss",
                                     save_last= True),
                ])
        
        train_dataloader = DataLoader(data_path=DATA_PATH, mode='train', batch_size=BATCH_SIZE, transform=transform_augmentation if DA else transform)
        val_dataloader = DataLoader(data_path=DATA_PATH, mode='test', batch_size=BATCH_SIZE, transform=transform)   
        os.makedirs(f"{tb_logger.log_dir}/Samples", exist_ok=True)
        os.makedirs(f"{tb_logger.log_dir}/Reconstructions", exist_ok=True)

        print(f"======= Training {models_setup['name']} =======")
        runner.fit(experiment, train_dataloader, val_dataloader)
        
    else:
        plot_history(   
            ae.training_routine(
                range(start,50),
                DataLoader(data_path=DATA_PATH, mode='train', batch_size=BATCH_SIZE, transform=transform_augmentation if DA else transform),
                DataLoader(data_path=DATA_PATH, mode='test', batch_size=BATCH_SIZE, transform=transform),
                os.path.join(DATA_PATH, "checkpoints/")
            )
        )
    
if __name__ == '__main__':
    main()