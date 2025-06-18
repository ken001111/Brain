import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# 예측 결과 로드
pred = nib.load('eye_001.nii.gz')  # 실제 파일명으로 변경
pred_data = pred.get_fdata()

# 중간 슬라이스 시각화
mid_slice = pred_data.shape[2] // 2

plt.figure(figsize=(15, 5))

# 여러 슬라이스 보기
for i in range(3):
    plt.subplot(1, 3, i+1)
    slice_idx = mid_slice - 5 + i*5
    plt.imshow(pred_data[:, :, slice_idx], cmap='gray')
    plt.title(f'Slice {slice_idx}')
    plt.colorbar()

plt.tight_layout()
plt.savefig('prediction_result.png')
print("Image saved as prediction_result.png")

# 3D 정보 출력
print(f"\n3D volume shape: {pred_data.shape}")
print(f"Eye volume: {(pred_data == 1).sum()} voxels")
print(f"Eye percentage: {(pred_data == 1).sum() / pred_data.size * 100:.2f}%")