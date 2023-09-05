import os
import nibabel as nib
import numpy as np
import torch

from ConvAE.model import AE
from ConvAE.utils import load_opt_params
from ConvAE.config import KEYS

from utils.testing import (generate_testing_set,
                           compute_correlation_results,
                           plot_correlation_results,
                           display_image, display_difference
                           )
from utils.img_alteration import alter_image
from utils.preprocess import transform_aug

DATA_PATH = 'data/Kaggle/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    prepro_path = os.path.join(DATA_PATH, "preprocessed")
    transform, _ = transform_aug()
    optimal_parameters = load_opt_params(prepro_path)
    ae = AE(keys=KEYS, **optimal_parameters).to(device)
    ae.load_checkpoint(data_path=DATA_PATH, eval=True)

    generate_testing_set(ae=ae, data_path= DATA_PATH, alter_image = alter_image, transform = transform)

    results = compute_correlation_results(data_path=DATA_PATH, measures='both')

    np.save(os.path.join(DATA_PATH, 'results.npy'), results)

    plot_correlation_results(results)

    test_ids = np.load(os.path.join(DATA_PATH,'saved_ids.npy'), allow_pickle=True).item().get('test_ids')
    patient_id = int(np.random.choice(test_ids, size = 1))
    print(f"Selected patient is Patient N°{patient_id:03d}")

    prediction = nib.load(os.path.join(DATA_PATH, "structured/patient{:03d}/mask.nii.gz".format(patient_id))).get_fdata()[:,:,:].transpose(2, 1, 0)
    reconstruction = np.round(nib.load(os.path.join(DATA_PATH,"reconstructions/patient{:03d}/mask.nii.gz".format(patient_id))).get_fdata().transpose(2, 1, 0),2)

    mid_frame = np.argmax([np.average(prediction[i]) for i in range (prediction.shape[0] -1 )])

    display_image(prediction[mid_frame], f'pred_patient_{patient_id:03d}.jpg')
    display_image(reconstruction[mid_frame], f'reconst_patient_{patient_id:03d};jpg')
    display_difference(reconstruction[mid_frame], prediction[mid_frame], f'diff_patient_{patient_id:03d}.jpg')
    

if __name__ == "__main__":
    main()

