import os
import nibabel as nib
import numpy as np
import torch

from utils.testing import display_image, display_difference

'''
Line arguments to implement:
    - data path, str
    - patient id
'''

DATA_PATH = 'data/liver/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    
    test_ids = np.load(os.path.join(DATA_PATH,'saved_ids.npy'), allow_pickle=True).item().get('test_ids')
    
    PATIENT_ID = int(np.random.choice(test_ids))
    print(f"Selected patient is Patient N°{PATIENT_ID:03d}")

    prediction = nib.load(os.path.join(DATA_PATH, "structured/patient{:03d}/mask.nii.gz".format(PATIENT_ID))).get_fdata()[:,:,:].transpose(2, 1, 0)
    reconstruction = np.round(nib.load(os.path.join(DATA_PATH,"reconstructions/patient{:03d}/mask.nii.gz".format(PATIENT_ID))).get_fdata().transpose(2, 1, 0),2)

    mid_frame = np.argmax([np.average(prediction[i]) for i in range (prediction.shape[0] -1 )])

    display_image(prediction[mid_frame], PATIENT_ID, f'pred_patient_{PATIENT_ID:03d}.png')
    display_image(reconstruction[mid_frame], PATIENT_ID, f'reconst_patient_{PATIENT_ID:03d}.png')
    display_difference(reconstruction[mid_frame], prediction[mid_frame], PATIENT_ID, f'diff_patient_{PATIENT_ID:03d}.png')
    

if __name__ == "__main__":
    main()

