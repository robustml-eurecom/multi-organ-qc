import os
import numpy as np
import torch
import yaml
import nibabel as nib

from utils.testing import display_image, display_difference
from utils.preprocess import transform_aug, find_segmentations, structure_dataset

from models.ConvAE.cae import ConvAutoencoder
from models.utils import load_opt_params

'''
Line arguments to implement:
    - data path, str
    - patient id
'''
organ = 'prostate'
DATA_PATH = os.path.join("data", organ)
PRED_PATH = os.path.join(DATA_PATH, "measures")
CONFIG_FILENAME = "multi_organ_qc/models/config.yml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    
    with open(CONFIG_FILENAME, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
         
    if not os.path.exists(f'{PRED_PATH}_structured'):
        segmentations_paths = find_segmentations(
                root_dir=PRED_PATH, 
                keywords=["patient"]
                )  
        structure_dataset(
                data_path = DATA_PATH, 
                mask_paths=segmentations_paths, 
                maskName="mask.nii.gz", 
                destination_folder='measures_structured',
                delete=PRED_PATH)
    
    test_ids = np.load(os.path.join(DATA_PATH,'saved_ids.npy'), allow_pickle=True).item().get('test_ids')
    
    PATIENT_ID = int(np.random.choice(test_ids))
    print(f"Selected patient is Patient N°{PATIENT_ID:03d}")

    prediction = nib.load(os.path.join(DATA_PATH, "measures_structured/patient{:03d}/mask.nii.gz".format(PATIENT_ID))).get_fdata()[:,:,:].transpose(2, 1, 0)
    reconstruction = np.round(nib.load(os.path.join(DATA_PATH,"reconstructions/patient{:03d}/mask.nii.gz".format(PATIENT_ID))).get_fdata().transpose(2, 1, 0),2)
    gt = nib.load(os.path.join(DATA_PATH, "structured/patient{:03d}/mask.nii.gz".format(PATIENT_ID))).get_fdata().transpose(2, 1, 0)
    mid_frame = np.argmax([np.average(prediction[i]) for i in range (prediction.shape[0] -1 )])

    display_image(prediction, PATIENT_ID, 'prediction')
    display_image(reconstruction, PATIENT_ID, 'reconstruction')
    
    display_image(gt, PATIENT_ID, 'gt', return_path=False)
    display_difference(img_rec, img_pred, PATIENT_ID, 'aberration_mask')
    

if __name__ == "__main__":
    main()

