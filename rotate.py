import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import os

def find_eye_centers(mask_data):
    labeled_3d = label(mask_data > 0)
    regions_3d = regionprops(labeled_3d)
    print(f"발견된 3D 영역 수: {len(regions_3d)}")

    centers = []
    for i, region in enumerate(regions_3d):
        z_center, y_center, x_center = region.centroid
        centers.append((x_center, y_center))
        print(f"영역 {i+1} 중심점: x={x_center:.1f}, y={y_center:.1f}, z={z_center:.1f}")

    centers_sorted = sorted(centers, key=lambda c: c[0])
    return centers_sorted

def calculate_rotation_angle(centers):
    if len(centers) < 2:
        print("두 개의 눈을 찾을 수 없습니다.")
        return 0

    left_eye, right_eye = centers
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)

    print(f"\n왼쪽 눈: ({left_eye[0]:.1f}, {left_eye[1]:.1f})")
    print(f"오른쪽 눈: ({right_eye[0]:.1f}, {right_eye[1]:.1f})")
    print(f"dx: {dx:.1f}, dy: {dy:.1f}")
    print(f"현재 기울기 각도: {angle_deg:.2f}도")
    print(f"회전할 각도: {-angle_deg:.2f}도")

    #angle_deg = 0 - angle_deg
    return angle_deg

def rotate_volume(volume_data, angle_deg, order=1):
    rotated = ndimage.rotate(
        volume_data,
        angle_deg,
        axes=(1, 2),
        reshape=False,
        order=order,
        mode='constant',
        cval=volume_data.min()
    )
    return rotated

def visualize_result(ct_data, mask_data, rotated_ct, rotated_mask, angle):
    mask_slices = np.where(mask_data.sum(axis=(1, 2)) > 0)[0]
    selected_slices = mask_slices[::max(1, len(mask_slices)//4)][:4]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, s in enumerate(selected_slices):
        axes[0, i].imshow(ct_data[s], cmap='gray', aspect='equal')
        axes[0, i].imshow(mask_data[s], cmap='Reds', alpha=0.5)
        axes[0, i].set_title(f'Original (slice {s})')
        axes[0, i].axis('off')

        axes[1, i].imshow(rotated_ct[s], cmap='gray', aspect='equal')
        axes[1, i].imshow(rotated_mask[s], cmap='Reds', alpha=0.5)
        axes[1, i].set_title(f'Rotated {angle:.1f}°')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('rotation_alignment_result.png', dpi=150, bbox_inches='tight')
    plt.close()

    center_slice = mask_slices[len(mask_slices)//2]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(ct_data[center_slice], cmap='gray', aspect='equal')
    axes[0].imshow(mask_data[center_slice], cmap='Reds', alpha=0.5)
    axes[0].set_title(f'Original (slice {center_slice})')
    axes[0].axis('off')

    axes[1].imshow(rotated_ct[center_slice], cmap='gray', aspect='equal')
    axes[1].imshow(rotated_mask[center_slice], cmap='Reds', alpha=0.5)
    axes[1].set_title(f'Rotated {angle:.1f}°')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('rotation_detail.png', dpi=150, bbox_inches='tight')
    plt.close()

def save_single_slice_comparison(ct_data, rotated_ct, save_path='slice_comparison.png'):
    z = ct_data.shape[0] // 2
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(ct_data[z], cmap='gray', aspect='equal')
    axes[0].set_title("Original Slice")
    axes[0].axis('off')

    axes[1].imshow(rotated_ct[z], cmap='gray', aspect='equal')
    axes[1].set_title("Rotated Slice")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"단일 슬라이스 비교 이미지 저장됨: {save_path}")

def main(ct_path, mask_path, output_ct_path, output_mask_path):
    import nibabel as nib
    print("=== 눈 정렬을 위한 CT 회전 ===\n")

    print("1. 데이터 로드 중...")
    ct_nii = nib.load(ct_path)
    mask_nii = nib.load(mask_path)
    ct_data = ct_nii.get_fdata()
    mask_data = mask_nii.get_fdata()
    print(f"   CT shape: {ct_data.shape}")
    print(f"   Mask shape: {mask_data.shape}")

    # 1-1. (H, W, Z) → (Z, H, W)
    ct_data = np.transpose(ct_data, (2, 1, 0))
    mask_data = np.transpose(mask_data, (2, 1, 0))

    print("\n2. 눈 중심점 찾기...")
    centers = find_eye_centers(mask_data)

    print("\n3. 회전 각도 계산...")
    angle = calculate_rotation_angle(centers)
    if abs(angle) < 0.5:
        print("\n회전 각도가 매우 작아 회전하지 않습니다.")
        return ct_data, mask_data, 0

    print(f"\n4. {angle:.2f}도 회전 수행 중...")
    rotated_ct = rotate_volume(ct_data, angle, order=1)
    rotated_mask = rotate_volume(mask_data, angle, order=0)

    print("\n5. 결과 시각화...")
    visualize_result(ct_data, mask_data, rotated_ct, rotated_mask, angle)
    save_single_slice_comparison(ct_data, rotated_ct)

    print("\n6. 결과 저장 중...")
    os.makedirs(os.path.dirname(output_ct_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)

    # 저장을 위해 다시 (512, 512, 50)로 transpose
    rotated_ct = np.transpose(rotated_ct, (1, 2, 0))
    rotated_mask = np.transpose(rotated_mask, (1, 2, 0))

    rotated_ct_nii = nib.Nifti1Image(rotated_ct, ct_nii.affine, ct_nii.header)
    rotated_mask_nii = nib.Nifti1Image(rotated_mask, mask_nii.affine, mask_nii.header)
    nib.save(rotated_ct_nii, output_ct_path)
    nib.save(rotated_mask_nii, output_mask_path)

    print(f"\n완료!")
    print(f"저장된 파일:\n  - CT: {output_ct_path}\n  - Mask: {output_mask_path}")
    print(f"생성된 이미지:\n  - rotation_alignment_result.png\n  - rotation_detail.png")

    return rotated_ct, rotated_mask, angle

# 실행
if __name__ == "__main__":
    ct_path = "nnUNet_raw/Dataset001_Eyeball/imagesTs/eye_001_0000.nii.gz"
    mask_path = "1000epoch test prediction/eye_001_clean.nii.gz"
    output_ct_path = "aligned_imagesTs/eye_001_0000_aligned.nii.gz"
    output_mask_path = "aligned_labels/eye_001_aligned.nii.gz"

    rotated_ct, rotated_mask, angle = main(
        ct_path, mask_path,
        output_ct_path, output_mask_path
    )