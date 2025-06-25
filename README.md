# 🧠 Eyeball Segmentation and Center Point Detection using nnUNet

## 📌 Project Overview
This project uses **nnU-Net** to train a 3D segmentation model that identifies the **eyeball regions** in head CT scans. After segmentation, we also compute the **center point of each eyeball** for potential use in downstream tasks such as gaze estimation or surgical navigation.

- 3D Slicer was used for manual annotation.
- The model was trained on **20 CT scans**, and inference was run on **7 CT scans**.
- Cross-validation with **5 folds** was applied during training.

---

## 🗂️ Folder Structure
```
project_root/
│
├── nnUNet_raw/
│   ├── Dataset001_Eyeball/
│   │   ├── imagesTr/          # 20 training CT volumes (NIfTI format)
│   │   ├── labelsTr/          # 20 eyeball segmentation masks
│   │   ├── imagesTs/          # 7 test CT volumes
│   │   └── dataset.json       # nnU-Net metadata
│
├── preprocess/
│   ├── convert_dicom_to_nifti.py
│   └── ...
│
├── results/
│   └── ...                    # Output predictions and metrics
│
└── README.md
```

---

## 🛠️ Setup & Installation

```bash
conda create -n nnunet python=3.10 -y
conda activate nnunet

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e "git+https://github.com/MIC-DKFZ/nnUNet.git#egg=nnunet"
```

Set environment variables:
```bash
export nnUNet_raw_data_base="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export RESULTS_FOLDER="/path/to/nnUNet_results"
```

---

## 🧾 Labeling Method

- Manual labeling was performed using **3D Slicer**.
- Both left and right eyeballs were annotated as a **single foreground class** (label: `1`).
- Labels were saved in **NIfTI (.nii.gz)** format and placed under `labelsTr/`.
- The **center point** of each eyeball was computed from the segmentation masks using a simple postprocessing step (e.g., centroid of connected components).

---

## ⚙️ Preprocessing

> **No preprocessing** such as windowing, interpolation, or cropping was applied. Raw CT volumes were used directly after conversion to NIfTI.

---

## 🧪 Training and Inference

### Training with 5-fold Cross Validation:
```bash
nnUNetv2_train 001 3d_fullres all
```

### Inference on test set:
```bash
nnUNetv2_predict -i /path/to/imagesTs \
                 -o /path/to/output \
                 -d 001 -c 3d_fullres -f 0
```

You can iterate `-f` from 0 to 4 for each fold if needed.

---

## 📊 Results
> To be updated after evaluation (e.g., Dice score for segmentation, average Euclidean distance for center point estimation).

---

## 📚 References

- [nnU-Net: Self-adapting Framework for U-Net-based Medical Image Segmentation](https://arxiv.org/abs/1809.10486)
- [3D Slicer](https://www.slicer.org/)
- [CQ500 Dataset](https://headctstudy.qure.ai/dataset)