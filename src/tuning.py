import sys, importlib
import os
import numpy as np
#importlib.reload(sys.modules['ConvAE'])
#importlib.reload(sys.modules['utils'])

from utils.dataset import train_val_test

from ConvAE.model import hyperparameter_tuning
from ConvAE.config import parameters, rules
from utils.preprocess import transform_aug
from utils.dataset import DataLoader

DATA_PATH = 'data/Kaggle'
OUTPUT_PATH = "optimal_parameters"

'''
List of args to be implemented:
    - data path / str
    - output path / str  
'''

def main():
    #train_ids, val_ids, _ = train_val_test(data_path=DATA_PATH)

    transform, transform_augmentation = transform_aug()
    
    train_ids = np.load(
        os.path.join(DATA_PATH, 'saved_ids.npy'),
        allow_pickle=True).item().get('train_ids')
    val_ids = np.load(
        os.path.join(DATA_PATH, 'saved_ids.npy'), 
        allow_pickle=True).item().get('val_ids')

    optimal_parameters = hyperparameter_tuning(
        parameters,
        DataLoader(
            os.path.join(DATA_PATH, "preprocessed/"), 
            patient_ids=train_ids[0:10], 
            batch_size=None, 
            transform=None),
        DataLoader(
            os.path.join(DATA_PATH, "preprocessed/"),
            patient_ids=val_ids,
            batch_size=None,
            transform=None),
        transform, 
        transform_augmentation,
        rules,
        fast=True)  #very important parameter. 
                    #When False, all combinations are tested to return the one retrieving the maximum DSC. 
                    #When True, the first combination avoiding dumb local minima is returned.
    np.save(os.path.join(os.path.join(DATA_PATH, "preprocessed/"), OUTPUT_PATH), optimal_parameters)
        

if __name__ == '__main__':
    main()