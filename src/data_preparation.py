import os
import numpy as np

from utils.preprocess import (
    generate_patient_info,
    preprocess,
    structure_dataset, 
    find_segmentations
    )

DATA_PATH = 'data/Kaggle/'
SEG_AREA = 'liver_tumor_segmentation'

'''
List of args to be implemented:
    - data path / str
    - seg_area / str
'''

def main():
    
    main_path = os.path.join(DATA_PATH, SEG_AREA)    
    segmentations_paths = find_segmentations(
        root_dir=main_path, 
        keywords=["segmentation"]
        )
    
    print(f"{len(segmentations_paths)} segmentations were successfully retrieved.")


    # The following folders will be deleted after structured_dataset is completed
    delete = [
        os.path.join(main_path,"volume_pt1/"),
        os.path.join(main_path,"volume_pt2/"),
        os.path.join(main_path,"volume_pt3/"),
        os.path.join(main_path,"volume_pt4/"),
        os.path.join(main_path,"volume_pt5/"),
        main_path
        ]

    structure_dataset(data_path = DATA_PATH, mask_paths=segmentations_paths, maskName="mask.nii.gz", delete=delete)

    _ = generate_patient_info(data_path=DATA_PATH, dataset_folder="structured/")
    preprocess(data_path=DATA_PATH)
    
if __name__ == '__main__':
    main()