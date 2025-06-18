import nibabel as nib
import numpy as np

# 파일 로드
pred = nib.load('eye_007.nii.gz')
pred_data = pred.get_fdata()

# 정확한 값 확인
unique_values = np.unique(pred_data)
print(f"Unique values: {unique_values}")
print(f"Data type: {pred_data.dtype}")

# 혹시 부동소수점 오차?
print(f"Values close to 1: {np.unique(pred_data[pred_data > 0.5])}")

# 정수로 변환 후 저장
pred_int = pred_data.astype(np.uint8)
new_nii = nib.Nifti1Image(pred_int, pred.affine, pred.header)
nib.save(new_nii, 'eye_007_clean.nii.gz')
print("Saved as eyeball_001_clean.nii.gz")