# 🧠 Eyeball Segmentation and Center Point Detection using nnUNet

## 📌 Project Overview
This project uses **nnU-Net** to segment the **eyeball regions** in head CT scans, with the primary goal of enabling **automated head alignment**.  
By extracting the center points of both eyes and rotating the image to horizontally align them, we achieve standardized orientation for downstream tasks such as gaze estimation, navigation, or anatomical normalization.

- Manual labels were created using **3D Slicer**
- Model trained on **20 CT scans**, tested on **7 CT scans**
- **5-fold cross-validation** was used during training
- Dataset source: [CQ500 Dataset](http://15.206.3.216/dataset)
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

- CT volumes used for training were preprocessed to standardize window size and slice spacing across samples.
- No other preprocessing (e.g., cropping or intensity normalization) was applied.

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

🔁 Post Processing

After model inference, we perform post-processing to extract meaningful anatomical information from the segmentation masks.

✔️ Center Point Extraction
	•	The center point of each eyeball is computed from the predicted mask using connected component analysis.
	•	For each connected component, we calculate the centroid using tools like:
	•	scipy.ndimage.center_of_mass
	•	or skimage.measure.regionprops

✔️ Rotation Based on Center Points
	•	After extracting both eyeball center points, the image is rotated so that:
	•	The line connecting the two centers becomes horizontal
	•	The head is properly aligned with the vertical midline
	•	This step ensures consistency and anatomical standardization across scans, which is especially important for alignment-sensitive tasks.

---

## 📊 Train Results
Below is the image of a rotated Brain CT, using the predicted eye mask. 
![Image](https://github.com/user-attachments/assets/fc01aa13-2363-4da8-a147-40e8a4f08996)
---

## 📚 References

- [nnU-Net: Self-adapting Framework for U-Net-based Medical Image Segmentation](https://arxiv.org/abs/1809.10486)
- [3D Slicer](https://www.slicer.org/)
- [CQ500 Dataset](http://15.206.3.216/dataset)