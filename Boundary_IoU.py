import numpy as np
import cv2
import os
import pandas as pd


def mask_to_boundary(mask, dilation_ratio=0.02):
    # Convert a binary mask to its boundary representation.
    image_diagonal = np.sqrt(mask.shape[0] ** 2 + mask.shape[1] ** 2)
    dilation_size = int(dilation_ratio * image_diagonal)

    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel)
    boundary = dilated_mask - mask
    return boundary


def boundary_iou(gt, dt, dilation_ratio=0.02):
    # Compute boundary IoU between two binary masks.
    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)

    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()

    if union < 1:
        return 0
    boundary_iou = intersection / union
    return boundary_iou


# Folder path
gt_folder = "Replace with the ground truth file path"
pt_folder = "Replace with the prediction truth file path"
output_excel_path = "Replace with your desired Excel save path"

# Get list of filenames
gt_files = sorted(os.listdir(gt_folder))
pt_files = sorted(os.listdir(pt_folder))

# Ensure the number of files in both folders is the same
if len(gt_files) != len(pt_files):
    raise ValueError("GT and PT folders must have the same number of images.")

# Initialize an empty DataFrame for saving results
results = pd.DataFrame(columns=["GT_File", "PT_File", "Boundary IoU"])

# Compute boundary IoU in batch
for gt_file, pt_file in zip(gt_files, pt_files):
    gt_image_path = os.path.join(gt_folder, gt_file)
    pt_image_path = os.path.join(pt_folder, pt_file)

    # Load image
    gt_image = cv2.imread(gt_image_path, cv2.IMREAD_GRAYSCALE)
    pt_image = cv2.imread(pt_image_path, cv2.IMREAD_GRAYSCALE)

    # Create a binary mask from the image (threshold = 128)
    _, gt_mask = cv2.threshold(gt_image, 128, 255, cv2.THRESH_BINARY)
    _, dt_mask = cv2.threshold(pt_image, 128, 255, cv2.THRESH_BINARY)

    # Compute boundary IoU
    iou = boundary_iou(gt_mask, dt_mask)

    # Append results to the DataFrame
    results = results.append({"GT_File": gt_file, "PT_File": pt_file, "Boundary IoU": iou}, ignore_index=True)

# Save the results to Excel
results.to_excel(output_excel_path, index=False)
print(f"Boundary IoU results saved to {output_excel_path}")