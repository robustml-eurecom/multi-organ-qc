# _Multi-organ QC_ : an attempt for a single modular package for multi-organ Quality Control on medical images

## Steps

### 1. Load your data
Upload medical images in .nii format for your organ study. 

_Suggestion_: skeleton should be the following

    📦data
    ┣ 📂organ1
    ┃ ┗ 📂segmentations
    ┗ 📂organ2
    ┃ ┗ 📂segmentations
    ┗ 📂...

Pay attention in letting each organ folder including a sub-folder named "segmentations": this allows the package to navigate properly between the generated masks, and do not creating misunderstandings.


_06-09-2023 Update: tested on Liver and Brain segmentations._


### 2. Preprocess data
To start processing and injecting the loaded data, run on your machine the following prompt:
 ```
 python one-to-rule-them-all/src/data_prepration.py
 ```
A series of displayables will notify you about as this step progresses.

At the end, the folder updates with the following structure:

    📦data
    ┗ 📂organ1
    ┃ ┣ 📂preprocessed
    ┃ ┃ ┣ 📜patient_info.npy
    ┃ ┃ ┣ 📜patient0_info.npy
    ┃ ┃ ┣ 📜...
    ┃ ┃ ┗ 📜patientN_info.npy
    ┃ ┗ 📂structured
    ┃ ┃ ┣ 📂patient0
    ┃ ┃ ┃ ┗ 📜mask.nii.gz
    ┃ ┃ ┣ 📂...
    ┃ ┃ ┗ 📂patientN
    ┃ ┃ ┃ ┗ 📜mask.nii.gz

This applies for each preprocessed organ. **NOTE**: during the preprocessing step, old files are deleted, hence consider keeping a backup folder.

_06-09-2023 Update: arguments must be declared in the aforementioned script before running. TODO: including line arguments._

### Step 3.a Fine tuning (optional)
Skip this if you already have an optimal parameter list to test, and see Step 3.b (there's a suggested hyperparameter list, but you can pass yours). Otherwise run the prompt:

```
python one-to-rule-them-all/src/tuning.py
```

### Step 3.b Training
Automatically detects the available gpu or cpu. This trains the AutoEncoder net for mask reconstruction. Checkpoints are saved in the chosen organ data path ``` data/organ/checkpoints ```. Run the prompt:

```
python one-to-rule-them-all/src/train.py
```

### Step 4 Testing & Evaluating
Test the AE performances on a test set. This saves the reconstructions files in a folder ``` data/organ/reconstructions``` to be used for evaluation. Images are in .nii.gz, and they follow the same skeleton provided in Step 2. To test, run:
```
python one-to-rule-them-all/src/test.py
```

In order to evaluate, simply run:

```
python one-to-rule-them-all/src/evaluate.py
```

After selecting a patient ID, the app will save in a dedicated folder (namely ```src/evaluations```) the following png images:

    📦evaluations
    ┣ 📂patient_ID
    ┃ ┣ 📜diff_patient_ID.png 
    ┃ ┣ 📜pred_patient_ID.png
    ┃ ┗ 📜reconst_patient_ID.png
    ┗ 📂...


Where _diff_patient_ID.png_ stands for the aberration mask after the QC check, _pred_patient_ID.png_ for the segmentations provided, and _reconst_patient_ID.png_ for the AE reconstruction.

_07-09-2023 Update_: 
* _Patient ID is internally randomically selected. TODO: pass it as a cmd arg_:
* _You can select IDs coming from the already processed test set. TODO: implement a full pipeline from out-of-data subjects._



