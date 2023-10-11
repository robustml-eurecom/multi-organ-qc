import os
import numpy as np

from utils.preprocess import (
    generate_patient_info,
    preprocess,
    structure_dataset, 
    find_segmentations, 
    find_pairs
    )

organ = 'spleen'
DATA_PATH = os.path.join("data", organ)
SEG_AREA = 'segmentations'
PAIR_FOLDER = False
IMAGE_PATHS = None

'''
List of args to be implemented:
    - data path / str
    - seg_area / str
    - pair_folder / bool
    - image_paths / bool
'''

def main():
    
    main_path = os.path.join(DATA_PATH, 'measures')
    
    if not PAIR_FOLDER:     
        segmentations_paths = find_segmentations(
            root_dir=main_path, 
            keywords=["segmentation"]
            )
    else: 
        segmentations_paths, IMAGE_PATHS= find_pairs(
            root_dir=main_path, 
            mask_keywords=['mask'], 
            image_keywords=['image']
            )
    
    assert len(segmentations_paths) > 0, "No segmentations were found."
    
    print(f"{len(segmentations_paths)} segmentations were successfully retrieved.")

    # The following folders will be deleted after structured_dataset is completed
    delete = [main_path] 

    if not os.path.exists(os.path.join(main_path, "structured")):
        structure_dataset(
            data_path = DATA_PATH, 
            mask_paths=segmentations_paths, 
            maskName="mask.nii.gz", 
            delete=delete)

    _ = generate_patient_info(data_path=main_path, dataset_folder="structured/")
    preprocess(data_path=main_path)
    
if __name__ == '__main__':
    main()