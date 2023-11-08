import os
import numpy as np
import argparse

from utils.preprocess import (
    generate_patient_info,
    preprocess,
    structure_dataset, 
    find_segmentations, 
    find_pairs
    )

def main(args):
    DATA_PATH = os.path.join(args.data, args.organ) if args.organ else args.data
    main_path = os.path.join(args.data, args.organ, args.mask_folder) if args.organ else os.path.join(args.data, args.mask_folder)
    print("+-------------------------------------+")
    print(f'Running in the following path: {DATA_PATH}. Data will be retrieved from {main_path}.')  
    print("+-------------------------------------+")
    
    if not os.path.exists(os.path.join(DATA_PATH, args.output)):
        if not args.pair_folder:     
            segmentations_paths = find_segmentations(
                root_dir=main_path, 
                keywords=args.keyword
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
        structure_dataset(
            data_path =DATA_PATH, 
            mask_paths=segmentations_paths, 
            destination_folder=args.output,
            maskName="mask.nii.gz", 
            delete=delete)

    _ = generate_patient_info(data_path=DATA_PATH, dataset_folder=args.output)
    preprocess(data_path=DATA_PATH)
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Testing script for MOQC.')    
    # Add command-line arguments
    parser.add_argument('-d', '--data', type=str, 
                        default='data', help='Data folder.')
    parser.add_argument('-mf', '--mask_folder', type=str, 
                        default='labels', help='Masks folder.')
    parser.add_argument('-o', '--output', type=str,
                        default='structured/', help='Output folder of the structured dataset.')
    parser.add_argument('-pf', '--pair_folder', type=bool,
                        default=False, help='Enable pair folder.')
    parser.add_argument('-og', '--organ', type=str, help='Selected organ.')
    parser.add_argument('-k', '--keyword', type=list, help='Keyword to identify your segmentations.')
    parser.add_argument('--verbose', action='store_false', help='Enable verbose mode.')
    
    args = parser.parse_args()
    main(args)