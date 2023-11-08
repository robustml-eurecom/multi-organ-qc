import numpy as np
from medpy.metric import binary
import os
import nibabel as nib
from typing import Callable
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import pandas as pd

import torch
import torch.nn as nn
import torchvision
import seaborn as sns
from IPython.display import display
from batchgenerators.augmentations.utils import resize_segmentation
from PIL import Image
from scipy import stats

from utils.dataset import AddPadding, CenterCrop, OneHot, DataLoader
from utils.preprocess import preprocess, median_spacing_target

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_metrics(prediction, reference, keys: list = None):
    # NB : for the heart, keys = ["_RV", "_MYO", "_LV"]
    results = {}
    if keys == None:
        ref = np.copy(reference)
        pred = np.copy(prediction)
        
        try:
            results["dc"] = binary.dc(np.where(ref!=0, 1, 0), np.where(np.rint(pred)!=0, 1, 0))
        except:
            results["dc"] = 0
        try:
            results["hd"] = binary.hd(np.where(ref!=0, 1, 0), np.where(np.rint(pred)!=0, 1, 0))
        except:
            results["hd"] = np.nan
    else : 
        for c,key in enumerate(keys,start=1):
            ref = np.copy(reference)
            pred = np.copy(prediction)

            ref = ref if c==0 else np.where(ref!=c, 0, ref)
            pred = pred if c==0 else np.where(np.rint(pred)!=c, 0, pred)

            try:
                results["dc" + key] = binary.dc(np.where(ref!=0, 1, 0), np.where(np.rint(pred)!=0, 1, 0))
            except:
                results["dc" + key] = 0
            try:
                results["hd" + key] = binary.hd(np.where(ref!=0, 1, 0), np.where(np.rint(pred)!=0, 1, 0))
            except:
                results["hd" + key] = np.nan
    return results


#TODO : Generalise postprocessing
def postprocess_image(image, info, current_spacing):
    """
    Takes the input image along with some information, and applies reverse transformation from the preprocessing step.

    Parameters
    ----------
        image: np.array
            The generated output from the auto encoder
        info: dict
            The corresponding patient_info to that image
        current_spacing: list
            The current image spacing, which corresponds to the median_spacing_target() of the training.
    """

    shape = info["shape"]
    shape = shape if len(info["shape"]) == 3 else shape[:3]
    num_classes = image.shape[-1]
    postprocessed = np.zeros(shape)
    crop = info["crop"][:3]
    original_shape = postprocessed[crop].shape
    original_spacing = info["spacing"]
    tmp_shape = tuple(np.round(original_spacing[1:] / current_spacing[1:] * original_shape[:2]).astype(int)[::-1])
    image = np.argmax(image, axis=1)
    image = np.array([torchvision.transforms.Compose([
            AddPadding(tmp_shape), CenterCrop(tmp_shape), OneHot(num_classes=num_classes)
        ])(image)])
    image = resize_segmentation(image.transpose(1,3,2,0), image.shape[1:2]+original_shape,order=1)
    image = np.argmax(image, axis=0)
    postprocessed[crop] = image
    return postprocessed

def testing(ae, data_path:os.PathLike, 
            test_loader:DataLoader, 
            patient_info:dict='default', 
            folder_predictions:os.PathLike='default', #Used for computing results
            folder_out:os.PathLike='default', 
            current_spacing:list='default', 
            compute_results:bool = True):
    """
    For every patient in the DataLoader, loads it and runs the ae on it. \n
    Then, generates its reconstruction, runs postprocess on top and saves a .nii.gz version of it in folder_out/{patient_folder}.
    """

    patient_info = np.load(os.path.join(data_path,'preprocessed/patient_info.npy'), allow_pickle=True).item() if patient_info=='default' else patient_info
    current_spacing = median_spacing_target(os.path.join(data_path, 'preprocessed')) if current_spacing=='default' else current_spacing
    folder_predictions=os.path.join(data_path, 'structured') if folder_predictions==None else folder_predictions
    folder_out = os.path.join(data_path, 'reconstructions')if folder_out==None else folder_out
    ae.eval()
    with torch.no_grad():
        results = {}
        for i, batch in tqdm(enumerate(test_loader), desc="Evaluating testing images: "):
            id = i
            prediction, reconstruction = [], []
            #for batch in patient: 
            batch = {"prediction": batch.to(device)}
            batch["reconstruction"], _ = ae.forward(batch["prediction"])
            prediction = torch.cat([prediction, batch["prediction"]], dim=0) if len(prediction)>0 else batch["prediction"]
            reconstruction = torch.cat([reconstruction, batch["reconstruction"]], dim=0) if len(reconstruction)>0 else batch["reconstruction"]
            prediction = prediction.argmax().cpu().numpy(),
            reconstruction = reconstruction.cpu().numpy()
            #reconstruction = postprocess_image(reconstruction, patient_info[id], current_spacing)

            folder_out_patient = os.path.join(folder_out, "patient{:03d}".format(id))
            if compute_results :
                results["patient{:03d}".format(id)] = evaluate_metrics(
                    nib.load(os.path.join(folder_predictions, f"patient{id:03d}", "mask.nii.gz")).get_fdata(),
                    reconstruction,
                    keys = None
                )
            if not os.path.exists(folder_out_patient) : os.makedirs(folder_out_patient)
            
            nib.save(
                nib.Nifti1Image(reconstruction, patient_info[id]["affine"], patient_info[id]["header"]),
                os.path.join(folder_out_patient,'mask.nii.gz')
            )
    return results


def generate_testing_set(ae:nn.Module , data_path:os.PathLike, alter_image:Callable, transform:torchvision.transforms.Compose, test_ids:list='default', opt_params:dict='default' ):
    """
        Given an alter_image function:
        - Will generate an altered nii file of the original mask in {data_path}/measures/structured_model
        - Will then save the preprocessed altered patients in {data_path}/measures/preprocessed_model
        - Will last run the model on the preprocessed_model .npy altered segmentation files to produce model_pGT in {data_path}/measures/pGT

        Parameters
        ----------
            ae: AE
                The auto_encoder, already loaded and set to eval
            data_path: os.PathLike
                The data_path (eg. 'data/brain')
            alter_image:Callable
                The inputed model
            transform:torchvision.transforms.Compose
                The set of transformations to apply
            test_ids: list (default='default')
                If set to default, will automatically load the test_ids from DATA_PATH/saved_ids.npy

    """
    # Creating the required paths
    required_paths = ["measures", "measures/preprocessed_model"]
    for path in required_paths:
        full_path = os.path.join(data_path, path)
        if not os.path.exists(full_path): os.makedirs(full_path)
    
    # Gathering info about the patients and dataset
    patient_info = np.load(os.path.join(data_path,'preprocessed/patient_info.npy'), allow_pickle=True).item()
    spacing = median_spacing_target(os.path.join(data_path, "preprocessed"), 2)
    optimal_parameters = np.load(os.path.join(data_path, "preprocessed", "optimal_parameters.npy"), allow_pickle=True).item() if opt_params == 'default' else opt_params
    BATCH_SIZE = optimal_parameters["BATCH_SIZE"]
    test_ids = np.load(os.path.join(data_path, 'saved_ids.npy'), allow_pickle=True).item().get('test_ids') if test_ids == 'default' else test_ids

    # Creates nii.gz for the model_GT, and preprocesses that model_GT for the AE to run
    # Altered .nii.gz files located in measures/structured_model
    # Altered preprocessed images located in measures/preprocessed_model
    # See preprocess documentation for more details
    preprocess(
        data_path=data_path,
        verbose=False,
        alter_image=alter_image
        )
    
    test_loader = DataLoader(data_path, 
                             mode='custom',
                             test_ids=test_ids, 
                             root_dir=os.path.join(data_path,'measures/preprocessed_model'), 
                             batch_size=BATCH_SIZE, 
                             transform=transform)
    
    # Evaluates the model_GT and saves in (measures/preprocessed_model) with the trained AE
    _ = testing(
        ae=ae,
        test_loader=test_loader,
        patient_info=patient_info,
        folder_predictions=os.path.join(data_path, "measures/preprocessed_model"),
        folder_out=os.path.join(data_path, "measures/pGT"),
        current_spacing=spacing,
        compute_results=False)


class Count_nan():
    def __init__(self):
        self.actual_nan = 0
        self.spotted_CA = 0
        self.FP_CA = 0
        self.total = 0
      
    def __call__(self, df): 
        df_AE = df[[column for column in df.columns if "p" in column]]
        df_GT = df[[column for column in df.columns if "p" not in column]]
        check_AE = np.any(np.isnan(df_AE.values), axis=1)
        check_GT = np.any(np.isnan(df_GT.values), axis=1)

        self.actual_nan += np.sum(check_GT)
        self.spotted_CA += np.sum(np.logical_and(check_GT, check_AE))
        self.FP_CA += np.sum(np.logical_and(np.logical_not(check_GT), check_AE))
        self.total += np.sum(np.any(np.isnan(df.values), axis=1))
      
    def __str__(self):
        string = "Anomalies (DSC=0/HD=nan): {}\n".format(self.actual_nan)
        string += "Spotted by CA: {}\n".format(self.spotted_CA)
        string += "False Positive by CA: {}\n".format(self.FP_CA)
        string += "Total discarded from the next plots: {}".format(self.total)
        return string


def compute_correlation_results(data_path, model, test_ids = 'default', measures = 'both'):
    assert measures in ['hd', 'dc', 'both'],"Parameter measures must be 'hd', 'dc' or 'both'"
    test_ids = np.load(os.path.join(data_path, 'saved_ids.npy'), allow_pickle=True).item().get('test_ids') if test_ids == 'default' else test_ids

    # Finding out the measures used
    dc, hd = False, False
    if measures == 'hd' or measures == 'both' : hd = True
    if measures == 'dc' or measures == 'both' : dc = True
    
    # Initialize dataframe
    df_results = pd.DataFrame()
    count_nan = Count_nan()
    
    for id in tqdm(test_ids, desc="Generating results: "):
        # Retrieving the paths of all the images to compute results from
        path_GT = os.path.join(data_path, 'structured/patient{:03d}/mask.nii.gz'.format(id))
        path_model_GT = os.path.join(data_path, '{}/structured/patient{:03d}/mask.nii.gz'.format(model, id))
        path_model_pGT = os.path.join(data_path, '{}/reconstructions/patient{:03d}/mask.nii.gz'.format(model, id))
        
        # Retrieving the images
        model_GT = nib.load(path_model_GT).get_fdata()
        GT = nib.load(path_GT).get_fdata()
        model_pGT = nib.load(path_model_pGT).get_fdata().squeeze().transpose(1,2,0).argmax(axis=-1)
        
        # Preprocessing
        dim = GT.shape[:2]
        model_pGT = CenterCrop(dim)((AddPadding(dim)(model_pGT))).transpose(1,2,0)
        model_GT = CenterCrop(dim)((AddPadding(dim)(model_GT))).transpose(1,2,0)
        
        # Removing the time dimension
        if len(GT.shape) == 4: GT = GT[:, :, :, 0]
        if len(model_GT.shape) == 4: model_GT = model_GT[:, :, :, 0]
        if len(model_pGT.shape) == 4: model_pGT = model_pGT[:, :, :, 0]
        
        # Compute metrics for each class and store in dataframe
        for class_id in range(1, int(GT.max()) + 1):  # assuming class IDs are 0, 1, 2, ..., n
            mask = (GT == class_id)
            if dc:
                dc_value = binary.dc(mask, np.rint(model_GT) == class_id)
                df_results.loc[id, f'model_GT_K{class_id}_DC'] = dc_value
                dc_value = binary.dc(mask, np.rint(model_pGT) == class_id)
                df_results.loc[id, f'model_pGT_K{class_id}_DC'] = dc_value
            if hd:
                try: hd_value = binary.hd(mask, np.rint(model_GT) == class_id)
                except: hd_value = np.nan
                df_results.loc[id, f'model_GT_K{class_id}_HD'] = hd_value
                try: hd_value = binary.hd(mask, np.rint(model_pGT) == class_id)
                except: hd_value = np.nan
                df_results.loc[id, f'model_pGT_K{class_id}_HD'] = hd_value

    # Save results
    #Set the id of patients as index
    df_results.set_index(np.array(test_ids), inplace=True)
    df_results.to_csv(os.path.join(data_path, 'results.csv'))
    df_results = df_results.replace(0, np.nan)
    print(count_nan(df_results))
    df_results.dropna(inplace=True)
    
    return df_results


def plot_distribution(df_results, args):
    """
    This function plots the distribution of the DC and HD scores for each class.

    Parameters:
    df_results (pd.DataFrame): The DataFrame containing the results.

    Returns:
    None
    """
    # Get the number of classes from the column names
    num_classes = int((len(df_results.columns) / 4))

    for class_id in range(1, num_classes):
        # Extract the DC and HD values for this class
        dc_model_GT = df_results[f'model_GT_K{class_id}_DC']
        dc_model_pGT = df_results[f'model_pGT_K{class_id}_DC']
        hd_model_GT = df_results[f'model_GT_K{class_id}_HD']
        hd_model_pGT = df_results[f'model_pGT_K{class_id}_HD']

        # Create a figure with two subplots
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        # Plot the DC values for model_GT
        axs[0, 0].hist(dc_model_GT, bins=20, alpha=0.5, color='b')
        axs[0, 0].set_title(f'Class {class_id} model_GT DC')
        axs[0, 0].set_xlabel('DC')
        axs[0, 0].set_ylabel('Frequency')

        # Plot the DC values for model_pGT
        axs[0, 1].hist(dc_model_pGT, bins=20, alpha=0.5, color='r')
        axs[0, 1].set_title(f'Class {class_id} model_pGT DC')
        axs[0, 1].set_xlabel('DC')
        axs[0, 1].set_ylabel('Frequency')

        # Plot the HD values for model_GT
        axs[1, 0].hist(hd_model_GT, bins=20, alpha=0.5, color='b')
        axs[1, 0].set_title(f'Class {class_id} model_GT HD')
        axs[1, 0].set_xlabel('HD')
        axs[1, 0].set_ylabel('Frequency')

        # Plot the HD values for model_pGT
        axs[1, 1].hist(hd_model_pGT, bins=20, alpha=0.5, color='r')
        axs[1, 1].set_title(f'Class {class_id} model_pGT HD')
        axs[1, 1].set_xlabel('HD')
        axs[1, 1].set_ylabel('Frequency')

        # Show the figure
        fig.suptitle("Plotted Distribution")
        fig.tight_layout()
        plt.show()
        plt.savefig(f'logs/{args.model.lower()}_{args.segmentations.lower()}_{args.organ.lower()}_distribution_plot.jpg')


def plot_correlation(df_results, args):
    """
    This function plots the correlation for each metric (DC and HD) between each class's pGT and GT.

    Parameters:
    df_results (pd.DataFrame): The DataFrame containing the results.

    Returns:
    None
    """
    # Get the number of classes from the column names
    num_classes = int((len(df_results.columns) / 4))

    # Create a figure with subplots
    cols = 2 if num_classes < 2 else num_classes
    fig, axs = plt.subplots(2, cols, figsize=(10 * num_classes, 10))

    for class_id in range(num_classes):
        # Extract the DC and HD values for model_GT and model_pGT for this class
        dc_model_GT = df_results[f'model_GT_K{class_id+1}_DC']
        dc_model_pGT = df_results[f'model_pGT_K{class_id+1}_DC']
        hd_model_GT = df_results[f'model_GT_K{class_id+1}_HD']
        hd_model_pGT = df_results[f'model_pGT_K{class_id+1}_HD']

        # Plot the DC values
        axs[0, class_id].scatter(dc_model_GT, dc_model_pGT)
        try:
            corr, _ = stats.pearsonr(dc_model_GT, dc_model_pGT)
            axs[0, class_id].legend([f'Correlation: {corr:.2f}'])
        except: axs[0, class_id].legend('Too many anomalies in the results to compute DC correlation')
        axs[0, class_id].set_title(f'Class {class_id+1} DC')
        axs[0, class_id].set_xlabel('model_GT')
        axs[0, class_id].set_ylabel('model_pGT')

        # Plot the HD values
        axs[1, class_id].scatter(hd_model_GT, hd_model_pGT)
        try:
            corr, _ = stats.pearsonr(hd_model_GT, hd_model_pGT)
            axs[1, class_id].legend([f'Correlation: {corr:.2f}'])
        except: axs[1, class_id].legend('Too many anomalies in the results to compute HD correlation')
        axs[1, class_id].set_title(f'Class {class_id+1} HD')
        axs[1, class_id].set_xlabel('model_GT')
        axs[1, class_id].set_ylabel('model_pGT')

    # Show the figure
    plt.suptitle("Plotted Correlation")
    plt.tight_layout()
    plt.savefig(f'logs/{args.model.lower()}_{args.segmentations.lower()}_{args.organ.lower()}_correlation_plot.jpg')

  
def display_plots(plots):
    plt.rcParams['xtick.labelsize'] = 30#'x-large'
    plt.rcParams['ytick.labelsize'] = 30#'x-large'
    plt.rcParams['legend.fontsize'] = 30#'x-large'
    plt.rcParams['axes.labelsize'] = 30#'x-large'
    plt.rcParams['axes.titlesize'] = 35#'x-large'

    grid = np.zeros([700*2, 700*3, 4])

    for i,measure in enumerate(["DSC", "HD"]):
        for j,label in enumerate(["LV", "RV", "MYO"]):
            x = "GT_{}_{}".format(measure, label)
            y = "pGT_{}_{}".format(measure, label)
            limx = np.ceil(max(plots[x] + plots[x]) / 10)*10 if measure=="HD" else 1
            limy = np.ceil(max(plots[y] + plots[y]) / 10)*10 if measure=="HD" else 1

            correlation = stats.pearsonr(plots[x], plots[y])[0]

            fig,axis = plt.subplots(ncols=1, figsize=(7, 7), dpi=100)
            sns.scatterplot(data=plots, x=x, y=y, ax=axis, label="Ours: r={:.3f}".format(correlation), color="blue", s=50)
            plt.plot(np.linspace(0, limx), np.linspace(0, limx), '--', color="gray", linewidth=5)

            axis.set_xlabel(measure)
            axis.set_ylabel("p{}".format(measure))
            axis.set_xlim([0, max(limx, limy)])
            axis.set_ylim([0, max(limx, limy)])
            axis.set_title(label)

            plt.grid()
            plt.tight_layout()
            plt.savefig("tmp.png")
            plt.close(fig)

            grid[i*700:(i+1)*700, j*700:(j+1)*700, :] = np.asarray(Image.open("tmp.png"))

    os.remove("tmp.png")
    grid = Image.fromarray(grid.astype(np.uint8))
    display(grid.resize((900,600), resample=Image.LANCZOS))


def display_image(img, out_folder, name): 
    assert name is not None, "Choose a valid name for your save."
    classes = np.unique(img)[-1] if np.unique(img)[-1] > 0 else 1
    img_gray = (img * 255/classes).astype(np.uint8)
    cv2.imwrite(os.path.join(out_folder, name), img_gray)
    #Image.fromarray(img.astype(np.uint8)).save(f'{folder_out_img}/{name}')
  
  
def display_difference(prediction, reference, out_folder, name):
    class_ref, class_pred = np.unique(reference)[-1], np.unique(prediction)[-1]
    pred_gray = (prediction * 255/class_pred).astype(np.uint8)
    dim = pred_gray.shape[:2]
    ref_gray = CenterCrop(dim)((AddPadding(dim)(reference * 255/class_ref).astype(np.uint8))).transpose(1,2,0)
    diff = cv2.absdiff(pred_gray, ref_gray)
    # Create an all-zero image with the same shape
    red_diff_image = cv2.merge([np.zeros_like(diff), np.zeros_like(diff), diff])
    cv2.imwrite(os.path.join(out_folder, name), red_diff_image)