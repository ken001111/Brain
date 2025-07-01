import SimpleITK as sitk
import os
from pathlib import Path
from tqdm import tqdm

# 고정된 spacing 지정
FIXED_SPACING = (0.488, 0.488, 3.0)
TARGET_SHAPE = (50, 512, 512)

def resample_volume(image, out_shape=TARGET_SHAPE, interpolator=sitk.sitkBSpline, out_spacing=FIXED_SPACING):
    resample = sitk.ResampleImageFilter()
    resample.SetSize([out_shape[2], out_shape[1], out_shape[0]])  # (X, Y, Z)
    resample.SetOutputSpacing(out_spacing)
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetInterpolator(interpolator)
    return resample.Execute(image)

def resample_folder_to_shape(images_dir, labels_dir, output_images_dir, output_labels_dir, out_shape=TARGET_SHAPE, out_spacing=FIXED_SPACING):
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    image_files = sorted(Path(images_dir).glob("*.nii.gz"))

    for image_path in tqdm(image_files):
        image = sitk.ReadImage(str(image_path))
        label_path = Path(labels_dir) / image_path.name
        label = sitk.ReadImage(str(label_path))

        # Resample
        resampled_img = resample_volume(image, out_shape=out_shape, interpolator=sitk.sitkBSpline, out_spacing=out_spacing)
        resampled_lbl = resample_volume(label, out_shape=out_shape, interpolator=sitk.sitkNearestNeighbor, out_spacing=out_spacing)

        # Save
        sitk.WriteImage(resampled_img, str(Path(output_images_dir) / image_path.name))
        sitk.WriteImage(resampled_lbl, str(Path(output_labels_dir) / label_path.name))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--labels_dir', type=str, required=True)
    parser.add_argument('--output_images_dir', type=str, required=True)
    parser.add_argument('--output_labels_dir', type=str, required=True)
    args = parser.parse_args()

    resample_folder_to_shape(
        args.images_dir,
        args.labels_dir,
        args.output_images_dir,
        args.output_labels_dir,
        out_shape=TARGET_SHAPE,
        out_spacing=FIXED_SPACING
    )