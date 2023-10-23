import os
import numpy as np
import torch
import yaml
import nibabel as nib
import argparse

from utils.testing import display_image, display_difference, testing
from utils.preprocess import transform_aug, find_segmentations, structure_dataset
from utils.dataset import DataLoader

from models.ConvAE.cae import ConvAutoencoder
from models.utils import load_opt_params

parser = argparse.ArgumentParser(description='Evaluation script for MOQC.')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_id(patient, ids=None):
    if patient.isnumeric(): return [int(patient)]
    elif patient == 'iter': return ids
    elif patient.lstrip("-").isdigit(): return [int(np.random.choice(ids))]
    else: raise ValueError("Patient ID not valid. Must be a string. Select between 'iter', '-1' or a number as string.")
    
def main():
    parser.add_argument('-d', '--data', type=str, 
                        default='data', help='Data folder.')
    parser.add_argument('-cf', '--config_file', type=str, 
                        default='moqc/models/config.yml', help='Configuration file.')
    parser.add_argument('-p', '--patient', type=str, 
                        default='-1', help='Patient ID. Leave empty for random selection. It allows "iter" if you want multiple patients.')
    parser.add_argument('-l', '--load', type=bool,
                        default=False, help='Load preprocessed data.')
    parser.add_argument('-og', '--organ', type=str, help='Selected organ.')
    parser.add_argument('-seg', '--segmentations', type=str, help='Folder with model segmentations.')
    parser.add_argument('--verbose', action='store_false', help='Enable verbose mode.')

    args = parser.parse_args()

    DATA_PATH = os.path.join(args.data, args.organ)
    
    with open(args.config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    prepro_path = os.path.join(DATA_PATH, "preprocessed")
    optimal_parameters = load_opt_params(prepro_path)
    transform, _ = transform_aug(num_classes=optimal_parameters['in_channels'])
    eval_ids = range(len(os.listdir(os.path.join(DATA_PATH, f'{args.segmentations}/structured')))) #if args.load else np.load(os.path.join(DATA_PATH, 'saved_ids.npy'), allow_pickle=True).item().get('test_ids')
    
    if args.load: 
        ae = ConvAutoencoder(keys=config["run_params"]["keys"], 
                            **optimal_parameters
                            ).to(device)
        ae.load_checkpoint(data_path=DATA_PATH, eval=True)
        
        _ = testing(
            ae=ae, 
            data_path=DATA_PATH,
            test_loader=DataLoader(data_path=DATA_PATH, mode='custom', patient_ids=eval_ids,
                                root_dir=os.path.join(DATA_PATH, f'{args.segmentations}/preprocessed'),
                                batch_size=optimal_parameters['batch_size'], transform=transform),
            folder_predictions=os.path.join(DATA_PATH, f'{args.segmentations}/structured'),
            folder_out=os.path.join(DATA_PATH, f'{args.segmentations}/reconstructions'),
            compute_results=False)
        
    patient_ids = extract_id(args.patient, eval_ids)
    for patient_id in patient_ids:
        print(f"Selected patient is Patient N°{patient_id:03d}")

        prediction = nib.load(os.path.join(DATA_PATH, "{}/structured/patient{:03d}/mask.nii.gz".format(args.segmentations, patient_id))).get_fdata()
        reconstruction = nib.load(os.path.join(DATA_PATH,"{}/reconstructions/patient{:03d}/mask.nii.gz".format(args.segmentations, patient_id))).get_fdata()
        gt = nib.load(os.path.join(DATA_PATH, "structured/patient{:03d}/mask.nii.gz".format(patient_id))).get_fdata()

        out_folder = os.path.join(DATA_PATH, f'evaluations/patient_{patient_id:03d}')
        if not os.path.exists(out_folder): os.makedirs(out_folder)
        
        display_image(prediction, out_folder, 'prediction.png')
        display_image(reconstruction, out_folder, 'reconstruction.png')
        display_image(gt, out_folder, 'gt.png')
        display_difference(prediction, reconstruction, out_folder, 'aberration_mask.png')
    

if __name__ == "__main__":
    main()

