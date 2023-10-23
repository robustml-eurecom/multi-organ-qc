import os
import gzip
import shutil
import warnings
import numpy as np
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt

from typing import Callable
from utils.common import ignore_directories

import torchvision
from utils.dataset import AddPadding, CenterCrop, OneHot, ToTensor, MirrorTransform, SpatialTransform


from scipy.ndimage import binary_fill_holes
from batchgenerators.augmentations.utils import resize_segmentation

def find_segmentations(root_dir:os.PathLike, keywords: list, absolute: bool = False) -> list :
    """
    Returns a list of all the paths to segmentations matching the keywords list. 
    If absolute is True, then absolute paths are returned. Otherwise, relative paths are given.

    Parameters
    ----------
    root_dir : os.PathLike
        The path in which to search for segmentations.
    keywords : list
        The list of strings to search for a match.
    absolute : bool
        If True, absolute paths are returned.
    Returns
    -------
    np.ndarray : the list of segmentations
    """
    assert type(keywords) != str, "Parameter keywords must be a list of str."
    segmentations = [[]]
    cwd = os.getcwd() if absolute else ""
    for dirElement in os.listdir(root_dir):
        subPath = os.path.join(root_dir, dirElement)
        if os.path.isdir(subPath):
            segmentations.append(find_segmentations(subPath, keywords, absolute))
        else :
            for keyword in keywords:
                if keyword in dirElement and (dirElement.endswith(".nii.gz") or dirElement.endswith(".nii")):
                    path = os.path.join(cwd,root_dir, dirElement)
                    segmentations.append(path)
        
    return np.unique(np.hstack(segmentations))

def find_pairs(root_dir:os.PathLike, mask_keywords, image_keywords, absolute:bool=False):
    """
    This function returns a mask_paths, image_paths for all segmentations found in the root dir, where a pair image/mask is found.
    This avoids having to deal with unpaired masks and image in case of an incomplete dataset

    Parameters
    ----------
    root_dir:os.PathLike
        The path in which to search for segmentations.
    mask_keywords: list
        The list of strings to search for a match in masks.
    image_keywords: list
        The list of strings to search for a match in images.
    absolute: bool
        If True, absolute paths are returned.

    Returns
    -------
    Filtered paths for : 
    (masks, images)
    """
    mask_paths = find_segmentations(root_dir = root_dir, keywords = mask_keywords, absolute = absolute)
    image_paths = find_segmentations(root_dir = root_dir, keywords = image_keywords, absolute = absolute)
    ignore = ignore_directories(mask_paths, image_paths)
    filtered_masks = []
    for p in mask_paths:
        if not any ([i in p for i in ignore]):
            filtered_masks.append(p)
    filtered_images = []
    for p in image_paths:
        if not any ([i in p for i in ignore]):
            filtered_images.append(p)
    
    return filtered_masks, filtered_images

def gunzip_and_replace(filePath:str):
    """
    Gunzips the given file and removes the previous one. Output will be in the same directory, suffixed by the .gz file identifier.

    Parameters
    ----------

        filePath: str
            The string path of the file to gunzip.
    """
    with open(filePath, 'rb') as f_in:
        with gzip.open(filePath + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            f_out.close()
        f_in.close()
    os.remove(filePath)

def structure_dataset(data_path:str,
                      mask_paths:os.PathLike,     
                      image_paths: os.PathLike = None,
                      destination_folder:str = 'structured', 
                      maskName: str="mask.nii.gz", 
                      imageName: str="image.nii.gz", 
                      delete: list=None) -> None:
    """
    This method uniformizes the dataset so that all other functions work on the same directory architecture.
    All segmentations pointed by segmentation_paths will be moved to destination_folder/patientXXX/fileName

    Parameters
    ----------
        segmentation_paths: list
            A list of all the paths of the segmentations to be restructured, length must be <= 1000
        data_path: str
            The specific data directory (eg. data/brain)
        destination_folder: str
            The new root_folder of the dataset. Both relative and absolute paths are accepted
        fileName: str
            The name of the new segmentation files
        delete: list(str)
            If set to a string path or a list of string paths, then specified folders will be deleted. Default: None.

    Note
    ----
        This limit of 1000 is solely dued to how the names are formatted (here :03d ; see source code). \n
        The limit can be removed if the patient folder names are adjusted throughout the code.

    """
    for path in mask_paths :
        assert path.endswith("nii.gz") or path.endswith('.nii'), f"segmentation must be of type .nii or .nii.gz but {path} was given."
    if image_paths is not None:
        for path in image_paths :
            assert path.endswith("nii.gz") or path.endswith('.nii'), f"image must be of type .nii or .nii.gz but {path} was given."

    assert maskName.endswith(".nii.gz"), "`maskName` must end with .nii.gz"
    assert imageName.endswith(".nii.gz"), "`fileName` must end with .nii.gz"
    #assert(len(mask_paths) <=1000 ), "Dataset is too large. Make sure N <= 1000"

    destination_folder = os.path.join(data_path, destination_folder)

    os.makedirs(destination_folder) if not os.path.exists(destination_folder) else None
    
    mask_paths.sort()
    destination_folder = os.path.join(os.getcwd(), destination_folder)
    os.makedirs(destination_folder) if not os.path.exists(destination_folder) else None


    for i in tqdm(range(len(mask_paths)), desc= 'Mask prepro Progress bar'):
        mask_path = mask_paths[i]
        image_path = image_paths[i] if image_paths is not None else None
        convert_mask = True if mask_path.endswith(".nii") else False
        if image_path is not None: 
                convert_image = True if image_path.endswith(".nii") else False
        else : convert_image = False
        # Creating the patient folder
        patient_target_folder = os.path.join(destination_folder, "patient{:03d}".format(i))
        os.makedirs(patient_target_folder) if not os.path.exists(patient_target_folder) else None

        # Changing target names according to .nii or .nii.gz extensions
        target_name_mask = os.path.join(patient_target_folder, maskName[:-3]) if convert_mask else os.path.join(patient_target_folder, maskName)
        target_name_image = os.path.join(patient_target_folder, imageName[:-3]) if convert_image else os.path.join(patient_target_folder, imageName)

        # Moves elements from the initial path to the target path (by default in {data_path}/structured/{patient_folder})
        # Then converts it if needed
        
        os.rename(mask_path, target_name_mask)
        if convert_mask : gunzip_and_replace(target_name_mask)

        if image_paths is not None : os.rename(image_path, target_name_image)
        if convert_image: gunzip_and_replace(target_name_image)

    if delete is not None:
        delete = np.array([delete]) if type(delete) == str else delete # Making delete iterable in case it's a simple string
        for path in delete:
            try:
                shutil.rmtree(path, ignore_errors=False)
            except:
                print("Could not delete specified folder {}".format(path))


def crop_image(image):
    """
    Returns the sections of the image containing non zero values ; i.e the smallest 3D box containing actual segmentation.

    Parameters
    ----------
    image: array
        The three dimensional matrix representation of the segmentation

    """
    nonzero_mask = binary_fill_holes(image != 0)
    mask_voxel_coords = np.stack(np.where(nonzero_mask))
    minidx = np.min(mask_voxel_coords, axis=1)
    maxidx = np.max(mask_voxel_coords, axis=1) + 1
    resizer = tuple([slice(*i) for i in zip(minidx,maxidx)])
    return resizer

def generate_patient_info(data_path:str, 
                          dataset_folder:os.PathLike='structured',
                          fileName: str="mask.nii.gz", 
                          skip:list = [], 
                          verbose: bool = False, 
                          verbose_rate: int = 10):
    """
    Generates patient info from the structructured dataset in dataset_folder, based on the files named fileName.\n
    Saves and outputs a dictionary of patients informations, with patient_id as key and gathers some information (see Gathered information)

    Parameters
    ----------
        data_path: str
            The specific data directory (eg. data/brain)
        dataset_folder: str
            The root folder containing the structured dataset.
        fileName: str
            The generic filename of each mask ; it should be the same for every patient.
        skip: list
            The integer list of the ids to skip.
        verbose: bool
            Enables the progress display every verbose_rate units processed.
        verbose_rate: int
            The step with which progress is displayed ; by default every 10 units.
    
    Gathered information :
    ----------------------
        Shape of the original image: "shape",
        The resizer used for cropping: "crop",
        The original spacing: "spacing",
        The original image header: "header",
        The image affine: "affine".
    """

    assert verbose_rate > 0, f"The verbose rate should be positive, but verbose_rate = {verbose_rate} was passed."

    dataset_folder = os.path.join(data_path, dataset_folder)

    patients_list = sorted(os.listdir(dataset_folder))
    print(len(patients_list))
    if patients_list[-1] == "patient_info.npy" : patients_list.pop(-1)
    if patients_list[0] == "optimal_parameters.npy" : patients_list.pop(0)

    patient_ids = [i for i in range(len(patients_list))]
    for id in skip:
        patient_ids.remove(id) # removing absent images or masks
    patient_info = {}
    for id in tqdm(range(len(patient_ids)), desc = 'Generate info progress Bar'):
        patient_folder = os.path.join(dataset_folder, 'patient{:03d}'.format(id))
        image = nib.load(os.path.join(patient_folder, fileName))
        patient_info[id] = {} # Initialising the dict for specified id
        patient_info[id]["shape"] = image.get_fdata().shape
        patient_info[id]["crop"] = crop_image(image.get_fdata())
        #patient_info[id]["crop"] = image.get_fdata()

        patient_info[id]["spacing"] = image.header["pixdim"][[3,2,1]]
        patient_info[id]["header"] = image.header
        patient_info[id]["affine"] = image.affine
        if(id%verbose_rate == 0 and verbose) : 
            print("Just processed patient {PATIENT_ID:03d} out of {TOTAL:03d}".format(PATIENT_ID = id, TOTAL=len(patient_ids)))

    patient_info_folder = os.path.join(data_path, 'preprocessed')
    if not os.path.exists(patient_info_folder):
        os.makedirs(patient_info_folder)  
        np.save(os.path.join(patient_info_folder, "patient_info"), patient_info)

    return patient_info

def median_spacing_target(folder: str, round=2) -> list:
    """
    Returns a spacing target as the median of all patients' spacing, rounded to specified decimal.

    Parameters
    ----------
        folder: str
            The directory containing patient_info.npy
        round: int
            The number of decimals of the rounding
    """
    path = os.path.join(folder, "patient_info.npy")
    patient_info = np.load(path, allow_pickle=True).item()
    spacing_target = np.median(np.array([patient_info[key]["spacing"] for key in patient_info.keys()]), axis=0)
    spacing_target[0] = 1
    return np.round(spacing_target, round)

def preprocess_image(image, crop, spacing, spacing_target):
    """
    Crops the input image into the smallest block strictly containing all the segmentation (removing black edges), and changes
    spacing from spacing to spacing_target.
    Returns the modified image;

    Parameters
    ----------
        image: array
            The three dimensional matrix representation of the image
        crop: list
            The resizer, usually the output of crop_image()
        spacing: list
            The current image spacing
        spacing_target:
            The target image spacing
    """
    # Removing the unused time dimension
    if len(image.shape) == 4: image = image[:, :, :, 0] 
    if len(crop) == 4: crop = crop[:3] 

    image = image[crop].transpose(2,1,0)
    spacing_target[0] = spacing[0]
    new_shape = np.round(spacing / spacing_target * image.shape).astype(int)
    image = resize_segmentation(image, new_shape, order=1)
    return image, new_shape

def preprocess(data_path:str, 
               patient_ids: range='default', 
               patient_info:dict='default', 
               spacing_target:list='default', 
               alter_image:Callable=None, 
               skip: list = [], 
               verbose: bool = False
               ) -> None: 
    """
    Produces a .npy for each given patient, it calls process_image and places the output in a new structured tree with root folder_out.\n
    preprocess() also updates the patient_info[id]["processed_shape"] the reflect the new shape from the preprocess.\n
    - If patient_info is not found, new patient_info will be generated.\n
    - If alter_image is set, then new altered nii images will be generated into {data_path}/measures/structured_model.
    - If alter_image is set, preprocessed images will be saved in measures/preprocessed_model.

    Parameters:
    -----------
        patient_ids: range / list
            The ids of the patients to be preprocessed, 'default' option will retrieve information from patient_info.npy
        patient_info: dict
            The generated patient info from generate_patient_info()
        spacing_target: 
            The target spacing, usually given by spacing_target()
        get_fname: Callable[[], str]
            A function returning the custom name of the patients masks
        alter_image: Callable, optional
            By default None ; will apply the specified alteration to the image before it is saved.
        skip: list, optional
            A list of the ids to skip , optional
        verbose: bool, optional
            When True, will display progress every 10 patients preprocessed

    """
    folder_in = os.path.join(data_path, 'structured')
    folder_out = os.path.join(data_path, 'preprocessed'if alter_image==None else 'measures/preprocessed_model') 
    get_patient_folder = lambda folder, id: os.path.join(folder, 'patient{:03d}'.format(id))
    get_fname = lambda : "mask.nii.gz"

    # Getting patient_info
    if patient_info=='default' and os.path.isfile(os.path.join(data_path, 'preprocessed/patient_info.npy')):
        patient_info = np.load(os.path.join(data_path, 'preprocessed/patient_info.npy'), allow_pickle=True).item()
    elif patient_info=='default' and not os.path.isfile(os.path.join(data_path, 'preprocessed/patient_info.npy')):
        message="patient_info doesn't exist, generating new patient_info from data in {data_path}/structured"
        warnings.warn(message, UserWarning)
        patient_info = generate_patient_info(data_path)

    
    spacing_target = median_spacing_target(os.path.join(data_path, 'preprocessed')) if spacing_target=='default' else spacing_target

    if not os.path.exists(folder_out) : os.makedirs(folder_out)

    patient_ids = [i for i in list(patient_info.keys()) if i not in skip] if patient_ids =='default' else patient_ids

    for id in patient_ids:
        patient_folder = get_patient_folder(folder_in, id)
        images = []
        fname = get_fname()
        fname = os.path.join(patient_folder, fname)
        if(not os.path.isfile(fname)):
            continue
        
        sample = nib.load(fname).get_fdata().astype(int)
        if len(np.shape(sample)) == 4: sample = sample[:,:,:,0] # Removing the time dimension
        # Apply optional alteration (eg. a model ; used to test the AE) and saves it
        if alter_image is not None:
            for i in range(np.shape(sample)[2]):
                sample[:,:,i] = alter_image(sample[:,:,i])

            folder_out_patient = os.path.join(data_path,'measures/structured_model', f'patient{id:03d}')
            if not os.path.exists(folder_out_patient) : os.makedirs(folder_out_patient)

            nib.save(
                    nib.Nifti1Image(sample, patient_info[id]["affine"], patient_info[id]["header"]),
                    os.path.join(folder_out_patient,'mask.nii.gz')
                )
        ### End of Alteration
        
        image, processed_shape = preprocess_image(
            sample,
            patient_info[id]["crop"],
            patient_info[id]["spacing"],
            spacing_target
        )

        # If depth of image changes after processing, it should be recorded
        patient_info[id]["processed_shape"] = processed_shape
        images.append(image)
        images = np.vstack(images)
        np.save(os.path.join(folder_out, "patient{:03d}".format(id)), images.astype(np.float32))
        if(verbose and id%10 == 0) : 
            print("Finished processing patient {:03d}".format(id))
    np.save(os.path.join(folder_out, "patient_info"), patient_info)


def transform_aug(num_classes):
    transform = torchvision.transforms.Compose([
        AddPadding((256,256)),
        CenterCrop((256,256)),
        #OneHot(num_classes=num_classes),
        ToTensor()
    ])
    transform_augmentation = torchvision.transforms.Compose([
        MirrorTransform(),
        SpatialTransform(patch_size=(256,256), angle_x=(-np.pi/6,np.pi/6), scale=(0.7,1.4), random_crop=True),
        OneHot(num_classes=num_classes),
        ToTensor()
    ])
    
    return transform, transform_augmentation