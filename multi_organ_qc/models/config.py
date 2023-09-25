import torch

RUN = False

KEYS=['BK', 'K1', 'K2']

# This is a list of possible values being tested for each hyperparameter.
tuning_parameters = {
    "DA" : [True, False], #data augmentation
    "latent_size": [100, 500], #size of the latent space of the autoencoder
    "BATCH_SIZE": [8, 16, 4],
    "lr": [2e-4, 1e-4, 1e-3],
    "weight_decay": [1e-5],
    "tuning_epochs": [5, 10], #number of epochs each configuration is run for
    "functions": [["GDLoss", "MSELoss"], ["GDLoss"], ["BKGDLoss", "BKMSELoss"]], #list of loss functions to be evaluated. BK stands for "background", which is a predominant and not compulsory class (it can lead to a dumb local minimum retrieving totally black images).
    "settling_epochs_BKGDLoss": [10, 0], #during these epochs BK has half the weight of LV, RV and MYO in the evaluation of BKGDLoss
    "settling_epochs_BKMSELoss": [10, 0], #during these epochs BK has half the weight of LV, RV and MYO in the evaluation of BKMSELoss
}

parameters = {
    "BATCH_SIZE": 8,
    "DA": False,
    "in_channels": 4,
    "out_channels": 4,
    "latent_channels": 100,
    "activation": "leaky_relu",
    "optimizer": torch.optim.Adam,
    "lr": 2e-4,
    "weight_decay": 1e-5,
    "settling_epochs_BKGDLoss": 10,
    "settling_epochs_BKMSELoss": 0
}

#this is a list of rules cutting out some useless combinations of hyperparameters from the tuning process.
rules = [
    '"settling_epochs_BKGDLoss" == 0 or "BKGDLoss" in "functions"',
    '"settling_epochs_BKMSELoss" == 0 or "BKMSELoss" in "functions"',
    '"BKGDLoss" not in "functions" or "settling_epochs_BKGDLoss" <= "tuning_epochs"',
    '"BKMSELoss" not in "functions" or "settling_epochs_BKMSELoss" <= "tuning_epochs"',
]

models_setup = {
    "name": "BetaVAE",
    "params": {
        "in_channels": 4,
        "latent_dim": 1000,
        "beta":  6.,
        "gamma": 1.
    },
    "exp_params":{          
        "LR": 2e-4,
        "weight_decay": 0.0005,
        "scheduler_gamma": 0.95,
        "kld_weight": 0.0025,
        "manual_seed": 1265
    },
    "logging_params": {
        "save_dir": "logs/",
        "name": 'BetaVAE'
    }    
}