# _Multi-organ QC_ : Quality Control on medical images segmentations using AutoEncoders

## Overview
This repository contains the code to run a quality control (QC) on medical images segmentations. The QC is based on a deep learning approach, namely an AutoEncoder (AE), which is trained to reconstruct the input mask. The difference between the input mask and the reconstructed one is the QC check. Results about the various scores and _correlation_ will complete the comprehensive evaluation.

If you need a guided tutorial, please find it in the jupyter notebook ```notebooks/tutorial.ipynb```.

## :computer: Requirements
* _Python_ > 3.8
* _PyTorch_ 1.12.0+cu113 (or whatever will best fit your machine. Note that this can cause some issues with the CUDA version, so please check the PyTorch website for the best fitting version)
* _Torchvision_ 0.13.0+cu113 (same as above)
* _Numpy_
* _Matplotlib_
* _Nibabel_
* _Scikit-learn_
* _Scipy_
* _tqdm_

You can find other requirements in the ```requirements.txt``` file. 

## :pencil: Important notes on data loading and preprocessing

### 1. Loading
You can choose the folder structure that fits the most for your convenience. In our case labels are retrieved by Medical Segmentation Decathlon dataset (MSD), and the segmentations are stored in custom folders corresponding at each UNet to be tested. After running a custom script (more info in the `tutorial.ipynb`) the final structure **MUST** be the following:

    📦data
    ┣ 📂organ1
    ┃ ┗ 📂labels
    ┃ ┃ ┣ 📜organ1_xxx_slice_yyy_.nii.gz
    ┃ ┃ ┗ 📜...
    ┃ ┗ 📂unet_1
    ┃ ┃ ┗ 📂segmentations
    ┃ ┃   ┣ 📜organ1_xxx_slice_yyy_.nii.gz
    ┃ ┃   ┗ 📜...
    ┃ ┗ 📂unet_2
    ┃ ┃ ┗ 📂segmentations
    ┃ ┗ 📂...
    ┗ 📂organ2
    ┃ ┣ 📂unet_1
    ┃ ┣ 📂unet_2
    ┃ ┣ 📂...
    ┃ ┗ 📂labels
    ┗ 📂...

### 2. Preprocess data
**_Do not run it if you already have your segmentations as shown before_**. To start processing and injecting the loaded data, run on your machine the following prompt as the `tutorial` suggests:
 ```
 python moqc/data_prepration.py --flags
 ```
A series of displayables will notify you about the process progress.

At the end, the folder updates with the following structure:

    📦data
    ┗ 📂organ1
    ┃ ┣ 📂preprocessed
    ┃ ┃ ┣ 📜patient_info.npy
    ┃ ┃ ┣ 📜patient0000_info.npy
    ┃ ┃ ┣ 📜...
    ┃ ┃ ┗ 📜patientNNNN_info.npy
    ┃ ┗ 📂structured
    ┃ ┃ ┣ 📂patient0000
    ┃ ┃ ┃  ┗ 📜mask.nii.gz
    ┃ ┃ ┣ 📂...
    ┃ ┃ ┗ 📂patientNNNN
    ┃ ┗ 📂unet_1
    ┃ ┃ ┗ 📂structured
    ┃ ┃ ┗ 📂preprocessed
    ┃ ┗ 📂unet_2
    ┃   ┗ 📂structured
    ┃   ┗ 📂preprocessed
    ┗ 📂...
    

This applies for each preprocessed organ. 

**NOTE**: during the preprocessing step, old files (e.g., `data/organ1/labels` and `data/organ1/unet_N/segmentations`) are deleted, hence consider keeping a backup folder.

## Model training and evaluation

### (Optional) Fine tuning
Skip this if you already have an optimal parameter list to test  (there's a suggested hyperparameter list in ```models/config.py```, but you can pass yours). Otherwise run the prompt:

```
python moqc/tuning.py
```

### Training
Automatically detects the available gpu or cpu. This trains the AutoEncoder net for mask reconstruction. Checkpoints are saved in the chosen organ data path ``` data/organ/checkpoints/model ```. Run the prompt:

```
python moqc/train.py --flags
```
More info on `tutorial.ipynb`.

### Testing & Evaluating
Test the AE performances on a test set. This saves the reconstructions files in a folder ``` data/organ/reconstructions``` to be used for evaluation. Images are in .nii.gz, and they follow the same skeleton provided in the Data section. To test, run:
```
python moqc/test.py --flags
```

The evaluation instead is performed by running the model at inference time, trying to reconstruct the input UNet mask. Please refer to the `tutorial.ipynb` for more info. Run the prompt:

```
python moqc/evaluate.py --flags
```

After selecting a patient ID, the app will save in a dedicated folder (namely ```evaluations/```) the following png images:

    📦evaluations
    ┣ 📂patient_NNNN
    ┃ ┣ 📜aberration_mask.png
    ┃ ┣ 📜prediction.png 
    ┃ ┣ 📜reconstruction.png
    ┃ ┗ 📜ground_truth.png
    ┗ 📂...


Where _aberration_mask.png_ stands for the anomalies identified by the model (the difference between the input mask and the reconstructed one), _prediction.png_ is the input mask segmentation (given by the tested UNet), _reconstruction.png_ is the reconstructed mask, and _ground_truth.png_ is the ground truth mask (given by MSD in our case).

Depending on the activation or not of the parameter `--correlation`, the model would eventually output a _.csv_ file containing the scores results (Dice Score and Hussendorf Distance) and the correlation (_Person R^2_) between the reconstruction and ground truth given a specific input mask. Correlation plots are saved in `logs/` folder, and they are named accordingly by the organ, the AE model and the UNet model. The notebook in `notebooks/results_analysis.ipynb` can be used to further investigate the all set of results.

# :notebook: Relevant Updates
_Update_10_11_23_: MOQC is tested on nnUNet v1 segmentations only. The repositories `evaluations/` and `logs/` have results related just to that network.

# :question: Get it run...maybe 
Please, refer to the `tutorial.ipynb` for more info. And, as always, feel free to contact me for any question or suggestion by opening an issue if you find any bug or problem :smile:.





