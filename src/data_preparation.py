import os
import numpy as np

from utils.preprocess import (
    generate_patient_info,
    preprocess,
    structure_dataset, 
    find_segmentations, 
    find_pairs
    )

organ = 'liver'
DATA_PATH = os.path.join("data", organ)
SEG_AREA = 'segmentations'
PAIR_FOLDER = False
IMAGE_PATHS = None
TO_STRUCTURE = False
'''
List of args to be implemented:
    - data path / str
    - seg_area / str
    - pair_folder / bool
    - image_paths / bool
    - to_structure / bool
'''

def main():
    
    main_path = os.path.join(DATA_PATH, SEG_AREA)
    
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
    
    print(f"{len(segmentations_paths)} segmentations were successfully retrieved.")


    # The following folders will be deleted after structured_dataset is completed
    delete = [
        os.path.join(DATA_PATH,"volume_pt1/"),
        os.path.join(DATA_PATH,"volume_pt2/"),
        os.path.join(DATA_PATH,"volume_pt3/"),
        os.path.join(DATA_PATH,"volume_pt4/"),
        os.path.join(DATA_PATH,"volume_pt5/"),
        main_path
        ] if not PAIR_FOLDER else main_path

    if TO_STRUCTURE:
        structure_dataset(
            data_path = DATA_PATH, 
            mask_paths=segmentations_paths, 
            maskName="mask.nii.gz", 
            delete=delete)

    _ = generate_patient_info(data_path=DATA_PATH, dataset_folder="structured/")
    preprocess(data_path=DATA_PATH)
    
if __name__ == '__main__':
    main()