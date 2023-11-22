import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import glob
from pathlib import Path
import yaml
from pathlib import Path
from natsort import natsorted
from sklearn.model_selection import train_test_split
from batchgenerators.augmentations.spatial_transformations import augment_spatial


class SliceDetector(object):
    """
    A class used to detect and save non-empty slices from 3D medical images.

    Attributes
    ----------
    data_path : str
        a string representing the path to the directory containing the 3D images
    save_path : str
        a string representing the path to the directory where the slices will be saved
    inter_slice_range : tuple, optional
        a tuple containing two integers representing the start and end indices of the slices to be selected
    non_zero_thres : float, optional
        a float representing the threshold for considering a slice as non-empty

    Methods
    -------
    __call__():
        Processes the 3D images and saves the non-empty slices to the save_path directory.
    remove_non_common_files(dir1, dir2):
        Removes files that are not common between two directories.
    """

    def __init__(self, inter_slice_range=None, non_zero_thres=None):
        self.inter_slice_range = inter_slice_range
        self.non_zero_thres = non_zero_thres

    def __call__(self, data_path, save_path):
        assert os.path.exists(data_path), f"{data_path} doesn't exist"
        to_save = os.path.join(save_path) 
        if not os.path.exists(to_save): os.makedirs(to_save)
        
        for label_f in os.listdir(data_path):
            if (".nii" in label_f):
                img = nib.load(os.path.join(data_path, label_f))
                np_img = img.get_fdata()
                
                if self.inter_slice_range is not None:
                    start, end = self.inter_slice_range
                    selected_slices = np_img[:, :, start:end]
                else:
                    non_zero_slices = np.any(np_img, axis=(0, 1))
                    selected_slices = np_img[:, :, non_zero_slices]
                    
                for i,slice in enumerate(non_zero_slices):
                    if slice:
                        slice_img = np_img[:, :, i]
                        if self.non_zero_thres is None or np.sum(slice_img) / np.prod(slice_img.shape) > self.non_zero_thres:
                            nib.save(nib.Nifti1Image(
                                np.expand_dims(slice_img, -1), img.affine), 
                                     os.path.join(to_save, label_f.replace('.nii', '_slice_{:03d}.nii'.format(i))))
    @staticmethod
    def remove_non_common_files(dir1, dir2):
        """
        Removes files that are not common between two directories.

        Parameters
        ----------
        dir1 : str
            The path to the first directory.
        dir2 : str
            The path to the second directory.
        """
        files1 = set(os.listdir(dir1))
        files2 = set(os.listdir(dir2))
        non_common_files = files1.symmetric_difference(files2)
        for file in non_common_files:
            if file in files1:
                os.remove(os.path.join(dir1, file))
            if file in files2:
                os.remove(os.path.join(dir2, file))
        print(f"Removed {len(non_common_files)} non-common files.")
        assert len(os.listdir(dir1)) == len(os.listdir(dir2))
        assert os.listdir(dir1).sort() == os.listdir(dir2).sort()


def extract_organ_idx(last_idx_dict:dict, test_ids:list):
    class_test_ids = {class_name: [] for class_name in last_idx_dict.keys()}
    sorted_last_idx_dict = sorted(last_idx_dict.items(), key=lambda x: x[1])
    for i, (class_name, class_idx) in enumerate(sorted_last_idx_dict):
        start_idx = 0 if i == 0 else sorted_last_idx_dict[i - 1][1]
        end_idx = class_idx
        print(class_name, start_idx, end_idx)
        class_test_ids[class_name] = test_ids[start_idx:end_idx]
        
        #class_test_ids[class_name] = [test_id - start_idx for test_id in class_test_ids[class_name]]
    return class_test_ids


def loso_split(data_path:str, mask_folder:str, split=[0.8, 0.1, 0.1], Force=False):
    assert sum(split) == 1 , f"Given split doesn't sum to 1 (input was {split})"
    assert len(split) == 3 , f"Given split doesn't have the right length : {len(split)} was given instead of 3"
    assert all([0<=split[i]<=1 for i in range(len(split))]), f"Split values must all be between 0 and 1."
    
    print('Splitting data following Leave One Subject Out (LOSO) method.')
    files = glob.glob(os.path.join(data_path, mask_folder, '*'))
    # Group the files by the first code
    groups = {}
    for i, filename in enumerate(natsorted(files)):
        stem_file = Path(filename).stem.split('_slice_')
        code = stem_file[0]
        slice_id = int((Path(stem_file[1]).stem).replace(".nii", ""))
        if code not in groups: 
            groups[code] = {'indices': [], 'slice_id': []}
        groups[code]['indices'].append(i)
        groups[code]['slice_id'].append(slice_id)
    print('Found {} different patients.'.format(len(groups)))
    
    train_ids = []
    val_ids = []
    test_ids = []
    train_codes, test_codes = train_test_split(list(groups.keys()), test_size=split[1], shuffle=False)
    train_codes, val_codes = train_test_split(train_codes, test_size=split[2], shuffle=False)

    print('Assigning indices...')    
    train_ids = np.concatenate([groups[code]['indices'] for code in train_codes])
    val_ids = np.concatenate([groups[code]['indices'] for code in val_codes])
    test_ids = np.concatenate([groups[code]['indices'] for code in test_codes])
    
    saved_ids = {'train_ids': train_ids, 'val_ids': val_ids, 'test_ids': test_ids}
    saved_ids_path = os.path.join(data_path, 'saved_ids.npy')
    organ_codes_path = os.path.join(data_path, 'organ_codes.npy')
    
    if not os.path.exists(saved_ids_path) or Force == True: 
        np.save(saved_ids_path, saved_ids)
        np.save(organ_codes_path, groups)
        print('Saved indices to {}.'.format(saved_ids_path))       
        print('Saved patient codes to {}.'.format(organ_codes_path))
    else:
        raise FileExistsError("Ids already exist, to overwrite, use parameter 'Force=True'")
    print('+-------------------------------------+') 
    
def train_val_test(data_path:str, ids_range:range='default', split=[0.70, 0.15, 0.15], shuffle = True, Force=False):
    """
    Given the specified range, returns three shuffled (by default) arrays given the specified split.


    Parameters
    ----------
        data_path: str
            The specific data path (eg. data/brain)
        ids_range: range
            The range of ids to split
        split: list (length 3)
            The list of split values : [train, test, val]. Sum must be equal to one.
        shuffle: bool, default=True
            If set to True, will shuffle the Ids
        Force: bool, default=False
            If set to True, this function wil overwrite any previously existing saved ids.
    """

    assert sum(split) == 1 , f"Given split doesn't sum to 1 (input was {split})"
    assert len(split) == 3 , f"Given split doesn't have the right length : {len(split)} was given instead of 3"
    assert all([0<=split[i]<=1 for i in range(len(split))]), f"Split values must all be between 0 and 1."

    patient_info = np.load(os.path.join(data_path, 'preprocessed/patient_info.npy'), allow_pickle=True).item()
    ids_range = range(len(os.listdir(f'{data_path}/structured'))) if ids_range == 'default' else ids_range

    ids = [i for i in ids_range]
    np.random.shuffle(ids)

    train_limit, val_limit = np.array([np.floor(split[i]*len(ids_range)) for i in [0,1]], dtype=int)
    val_limit += train_limit

    train_ids = ids[:train_limit]
    val_ids = ids[train_limit:val_limit]
    test_ids = ids[val_limit:]

    saved_ids = {'train_ids': train_ids, 'val_ids': val_ids, 'test_ids': test_ids}
    saved_ids_path = os.path.join(data_path, 'saved_ids.npy')

    if not os.path.exists(saved_ids_path) or Force == True: 
        np.save(saved_ids_path, saved_ids)
    else:
        raise FileExistsError("Ids already exist, to overwrite, use parameter 'Force=True'")

    return train_ids, val_ids, test_ids

class AddPadding(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def resize_image_by_padding(self, image, new_shape, pad_value=0):
        if len(image.shape) == 3: image = image.squeeze()
        shape = tuple(list(image.shape))
        new_shape = tuple(np.max(
            np.concatenate((shape, new_shape)).reshape((2, len(shape))), 
            axis=0
            ))
        if pad_value is None:
            if len(shape) == 2:
                pad_value = image[0, 0]
            elif len(shape) == 3:
                pad_value = image[0, 0, 0]
            else:
                raise ValueError("Image must be either 2 or 3 dimensional")
        res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
        start = np.array(new_shape) / 2. - np.array(shape) / 2.
        if len(shape) == 2:
            res[int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1])] = image
        elif len(shape) == 3:
            res[int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1]),
            int(start[2]):int(start[2]) + int(shape[2])] = image
        return res
  
    def __call__(self, sample):
        sample = self.resize_image_by_padding(sample, new_shape=self.output_size)
        return sample

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def center_crop_2D_image(self, img, crop_size):
        if all(np.array(img.shape) <= crop_size):
            return img
        center = np.array(img.shape) / 2.
        if type(crop_size) not in (tuple, list):
            center_crop = [int(crop_size)] * len(img.shape)
        else:
            center_crop = crop_size
            assert len(center_crop) == len(img.shape)
        return img[int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.),int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.)]

    def __call__(self, sample):
        sample = self.center_crop_2D_image(sample, crop_size=self.output_size)
        return sample.reshape(1, sample.shape[0], sample.shape[1])

class OneHot(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        
    def one_hot(self, seg):
        one_hot_image = np.zeros((self.num_classes, seg.shape[1], seg.shape[2]), dtype=np.uint8)
        unique_values = np.linspace(0, 255, self.num_classes, dtype=np.uint8)
        value_to_channel = {k:v for v,k in enumerate(unique_values)}
        for value, channel in value_to_channel.items():
            one_hot_image[channel, ...] = (seg == value).astype(np.uint8)
        return one_hot_image
    
    def __call__(self, sample):
        sample = self.one_hot(sample)
        return sample

class ToTensor(object):
    def __call__(self, sample):
        sample = torch.from_numpy(sample).float()
        return sample

class MirrorTransform():
    def __call__(self, sample):
        if np.random.uniform() < 0.5:
            sample = np.copy(sample[::-1])
        if np.random.uniform() < 0.5:
            sample = np.copy(sample[:, ::-1])
        return sample

class SpatialTransform():
    def __init__(self, patch_size, do_elastic_deform=False, alpha=None, sigma=None,
        do_rotation=True, angle_x=(-np.pi/6,np.pi/6), angle_y=None, angle_z=None,
        do_scale=True, scale=(0.7, 1.4), border_mode_data='constant', border_cval_data=0, order_data=3,
        border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, p_el_per_sample=1,
        p_scale_per_sample=1, p_rot_per_sample=1,independent_scale_for_each_axis=False, p_rot_per_axis:float=1,
        p_independent_scale_per_axis: int=1):
        
        self.params = locals()
        self.params.pop("self")
        self.params["patch_center_dist_from_border"] = list(np.array(patch_size) // 2)

    def __call__(self, sample):
        sample = sample[None,None,:,:]
        _,sample = augment_spatial(sample, sample, **self.params) 
        return sample[0,0]
    
############################
#### Dataset and Loader ####
############################

class NiftiDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, mode='all', is_segment=False):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.is_segment = is_segment
        self.patients = []
        
        split_dir = self.root_dir.split('/')[:-2] if is_segment else self.root_dir.split('/')[:-1]
        organ = split_dir[-1]
        print("Getting {} ids from {}".format(mode, os.path.join("/".join(split_dir),'saved_ids.npy')))
        self.ids = np.load(
            os.path.join('/'.join(split_dir),'saved_ids.npy'), 
            allow_pickle=True).item().get(f'{mode}_ids') if self.mode != 'all' else None
        
        self.image_paths = self.get_image_paths()
        if self.mode == 'all': self.ids = list(range(len(self.image_paths)))
        
        self.organ_codes = np.load(
            os.path.join('/'.join(split_dir), 'organ_codes.npy'), 
            allow_pickle=True).item()
        
        with open(os.path.join('/'.join(split_dir), 'preprocessed_classes.yml'), 'r') as f: 
            identifiers = yaml.safe_load(f)
            self.identifiers = [v['id'] for v in identifiers.values()]
            print(f"Organs' identifiers are {self.identifiers}")
        
        assert len(organ.split('_')) == len(self.identifiers), f"Number of organs ({len(organ.split('_'))}) doesn't match number of identifiers ({len(self.identifiers)})"
        for identifier in self.identifiers:
            self.patients.append([{code:self.organ_codes[code]} for code in self.organ_codes.keys() 
                            if self.organ_codes[code]['indices'][0] in self.ids and code.split('_')[0]==identifier])        
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = nib.load(image_path).get_fdata()
        classes = np.unique(image)[-1]
        pil_image = (image * 255/classes).astype(np.uint8)
        if self.transform:
            image = self.transform(pil_image)
        return image

    def get_image_paths(self):
        image_paths = []
        for patient_folder in os.listdir(self.root_dir):
            if self.ids is not None:
                if int(patient_folder.replace("patient", "")) not in self.ids: continue
            patient_folder_path = os.path.join(self.root_dir, patient_folder)
            if os.path.isdir(patient_folder_path):
                nifti_file = os.path.join(patient_folder_path, "mask.nii.gz")
                if os.path.exists(nifti_file):
                    image_paths.append(nifti_file)
        return image_paths


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, data_path:str, mode:str, root_dir:str='default', patient_ids=None, batch_size=None, transform=None, num_workers=0):

        assert mode in ['train', 'val', 'test', 'custom'], "Make sure mode is either 'train', 'val', 'test' or 'custom"
        if mode == 'custom': assert patient_ids is not None, 'patient_ids must be specified on custom mode'
        if patient_ids is not None and mode in ['train', 'val', 'test'] : print(f"Specified patient_ids will be ignored since default mode '{mode}' is specified")

        self.data_path = data_path
        self.root_dir = os.path.join(data_path, 'preprocessed') if root_dir == 'default' else root_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers
        self.patient_loaders = []
        if mode in ['train', 'val', 'test']:
            self.patient_ids = np.load(os.path.join(data_path,'saved_ids.npy'), allow_pickle=True).item().get(f'{mode}_ids')
        else:
            self.patient_ids = patient_ids

        if batch_size is not None:
            for id in self.patient_ids:
                self.patient_loaders.append(torch.utils.data.DataLoader(
                    Patient(self.root_dir, id, transform=self.transform),
                    batch_size=self.batch_size, shuffle=False, num_workers=0
                ))
        
        self.counter_id = 0

    def set_batch_size(self, batch_size):
        self.patient_loaders = []
        for id in self.patient_ids:
            self.patient_loaders.append(torch.utils.data.DataLoader(
                Patient(self.root_dir, id, transform=self.transform),
                batch_size=batch_size, shuffle=False, num_workers=0
            ))
    
    def set_transform(self, transform):
        self.transform = transform
        for loader in self.patient_loaders:
            loader.dataset.transform = transform

    def __iter__(self):
        self.counter_iter = 0
        return self

    def __next__(self):
        if(self.counter_iter == len(self)):
            raise StopIteration
        loader = self.patient_loaders[self.counter_id]
        self.counter_id += 1
        self.counter_iter += 1
        if self.counter_id%len(self) == 0:
            self.counter_id = 0
        return loader

    def __len__(self):
        return len(self.patient_ids)

    def current_id(self):
        return self.patient_ids[self.counter_id]

class Patient(torch.utils.data.Dataset):
    """
    The Dataset class representing one patient.

    Attributes
    ----------
        id: int
            The id of the patient. It is a unique identifier
        info: dict
            The corresponding item from patient_info. See generate_patient_info for details
        transform: torchvision.transforms.transforms.Compose
            The transforms applied to the patient data

    Methods
    -------
        __len__
            Returns the number of layers if the data matrix
        __getitem__
            Takes the specified layer (by the slice_id parameter), applies transformation and returns it
        
    """

    def __init__(self, root_dir, patient_id, transform=None):
        self.root_dir = root_dir
        self.id = patient_id
        self.info = np.load(os.path.join(root_dir,"patient_info.npy"), allow_pickle=True).item()[patient_id]
        self.transform = transform

    def __len__(self):
        return self.info["processed_shape"][0]

    def __getitem__(self, slice_id):
        data = np.load(os.path.join(self.root_dir, "patient{:05d}.npy".format(self.id)))
        sample = data#[slice_id]
        if self.transform:
            sample = self.transform(sample)
        return sample
            
