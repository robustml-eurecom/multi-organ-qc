import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from batchgenerators.augmentations.spatial_transformations import augment_spatial


def select_valid_imgs(data_path:str, save_path:str, inter_slice_range:list=[4, 14], non_zero_thres:float=0.1):
    """
    Select and save valid slices from 3D MRI label and segmentation images as 2D NIfTI images.

    This function processes 3D MRI label and segmentation images, selecting valid slices based on a non-zero threshold
    and a specified range of slices. It then saves the selected slices as 2D NIfTI images.

    Args:
    data_path (str): Path to the directory containing NIfTI files.
    save_path (str): Path to the directory where the selected slices will be saved.
    inter_slice_range (list): A list specifying the range of slices to consider (inclusive). Default is [5, 14].
    non_zero_thres (float): The threshold for considering a slice as valid based on the sum of non-zero values.
                           Default is 0.1.

    Raises:
    AssertionError: If any of the provided paths do not exist, or the images are 2D-based (not 3D).

    Note:
    - Images should be 3D. Images with only 2D slices are not processed.
    - Valid slices are determined based on the non-zero threshold.

    """  
    assert os.path.exists(data_path), f"{data_path} doesn't exist"
    to_save = os.path.join(save_path) 
    if not os.path.exists(to_save): os.makedirs(to_save)
    
    for label_f in os.listdir(data_path):
        if (".nii" in label_f):
            max_slice = inter_slice_range[1]
            img = nib.load(os.path.join(data_path, label_f))
            new_img = img.get_fdata().transpose(2, 1, 0)
            assert img.shape[0] > 1, f"Label and segmentation shapes are 2D-based. Please select a folder containing 3D images."
            
            if new_img.shape[0] < inter_slice_range[1]: max_slice = new_img.shape[0]
            for slice in range(inter_slice_range[0], max_slice):
                #print(np.sum(new_img[slice])/np.prod(new_img[slice].shape))
                if non_zero_thres is None or np.sum(new_img[slice]) > non_zero_thres:
                    nib.save(nib.Nifti1Image(np.expand_dims(new_img[slice], -1), img.affine), os.path.join(to_save, label_f.replace('.nii', f'_slice_{slice}.nii')))
    
    
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
    ids_range = range(len(patient_info)) if ids_range == 'default' else ids_range

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
        new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2, len(shape))), axis=0))
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
        return sample

class OneHot(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
    def one_hot(self, seg):
        try:
            return F.one_hot(torch.tensor(seg).long(), num_classes=self.num_classes)
        except:
            return F.one_hot(torch.tensor(np.where(seg > .5, 1, 0)).long(), num_classes=self.num_classes)
    def __call__(self, sample):
        if self.num_classes < 3: sample = np.where(sample > 0, 1, 0)
        sample = self.one_hot(sample).cpu().numpy() 
        return sample.transpose(2,0,1)

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
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self.get_image_paths()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = nib.load(image_path).get_fdata()
        # Convert the NIfTI array to a PIL image
        pil_image = (image * 255).astype(np.uint8)
        if self.transform:
            image = self.transform(pil_image)
        
        return image

    def get_image_paths(self):
        image_paths = []
        for patient_folder in os.listdir(self.root_dir):
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
        data = np.load(os.path.join(self.root_dir, "patient{:03d}.npy".format(self.id)))
        sample = data[slice_id]
        if self.transform:
            sample = self.transform(sample)
        return sample\
            
