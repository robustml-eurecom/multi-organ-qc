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

def main():
    parser.add_argument('-d', '--data', type=str, 
                        default='data', help='Data folder.')
    parser.add_argument('-cf', '--config_file', type=str, 
                        default='moqc/models/config.yml', help='Configuration file.')
    parser.add_argument('-p', '--patient', type=int, 
                        default=-1, help='Patient ID. Leave empty for random selection.')
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
    eval_ids = range(len(os.listdir(os.path.join(DATA_PATH, f'{args.segmentations}/structured'))))
    
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
        
    PATIENT_ID = int(np.random.choice(eval_ids)) if args.patient == -1 else int(args.patient)
    print(f"Selected patient is Patient N°{PATIENT_ID:03d}")

    prediction = nib.load(os.path.join(DATA_PATH, "{}/structured/patient{:03d}/mask.nii.gz".format(args.segmentations, PATIENT_ID))).get_fdata()[:,:,:].transpose(2, 1, 0)
    reconstruction = np.round(nib.load(os.path.join(DATA_PATH,"{}/reconstructions/patient{:03d}/mask.nii.gz".format(args.segmentations, PATIENT_ID))).get_fdata().transpose(2, 1, 0),2)
    gt = nib.load(os.path.join(DATA_PATH, "structured/patient{:03d}/mask.nii.gz".format(PATIENT_ID))).get_fdata().transpose(2, 1, 0)
    mid_frame = np.argmax([np.average(prediction[i]) for i in range (prediction.shape[0] -1 )])

    out_folder = os.path.join(DATA_PATH, f'evaluations/patient_{PATIENT_ID:03d}')
    if not os.path.exists(out_folder): os.makedirs(out_folder)
    
    #dummy approximation for better visualization
    #prediction = np.where(prediction < 0.5, 0, 1)
    display_image(prediction, out_folder, 'prediction.png')
    display_image(reconstruction, out_folder, 'reconstruction.png')
    display_image(gt, out_folder, 'gt.png')
    display_difference(prediction, reconstruction, out_folder, 'aberration_mask.png')
    

if __name__ == "__main__":
    main()

