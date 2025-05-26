import numpy as np
import os

import matplotlib.pyplot as plt
import tqdm

# 92 -7 = 85个
preprocess_dir = 'E:\LiverSegment\data\XiangYa'
# preprocess_dir = 'D:\data\XiangYa'
import SimpleITK as sitk
import os
import re

def find_patient_dirs(root_dir):
    patient_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if re.match(r'patient\d+$', dirname):
                full_path = os.path.join(dirpath, dirname)
                # 检查子目录是否包含patient+大写字母
                contains_subdir = False
                for subdir in os.listdir(full_path):
                    if os.path.isdir(os.path.join(full_path, subdir)) and re.match(r'patient\d+[A-Z]$', subdir):
                        contains_subdir = True
                        patient_dirs.append(os.path.join(full_path, subdir))
                        # break
                if not contains_subdir:
                    patient_dirs.append(full_path)
        # 由于os.walk递归遍历所有子目录，我们只需要检查顶层目录
        break
    return patient_dirs

def find_all_nii_gz_files(directory):
    nii_gz_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".nii.gz"):
                nii_gz_files.append(file)
    return nii_gz_files


def separate_segment_files(file_list):
    segment_files = []
    non_segment_files = []

    for file in file_list:
        if 'Segment' in file:
            segment_files.append(file)
        else:
            non_segment_files.append(file)

    return non_segment_files, segment_files

# 示例使用
root_dir = 'E:\LiverSegment\data\XiangYa_Raw'  # 请将此路径替换为你的文件目录路径
patient_dirs = find_patient_dirs(root_dir)
print(f"{len(patient_dirs)} 个样本")
# for dir in patient_dirs:
#     print(dir)

idx = 0 # 切片索引

for subject_path in tqdm.tqdm(patient_dirs):
    # 收集目录以nii.gz结尾的文件名
    nii_gz_files = find_all_nii_gz_files(subject_path)
    # 将标签和原始影像分开
    raw_files, segment_files = separate_segment_files(nii_gz_files)
    # 读取原始影像
    ct = sitk.ReadImage(os.path.join(subject_path, raw_files[0]), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    # 依次读取标签
    for segment_path in segment_files:
        seg = sitk.ReadImage(os.path.join(subject_path, segment_path), sitk.sitkInt8)
        seg_array = sitk.GetArrayFromImage(seg)
        z = np.any(seg_array, axis=(1, 2))  # 判断每一张切片中是否包含1（1表示肝脏），返回一个长度等于切片数的布尔数组
        try:
            start_slice, end_slice = np.where(z)[0][[0, -1]]  # np.where(z)返回数组中不为0的下标list
        except Exception as e :
            print(f"样本{subject_path}")
            print(f"标签{segment_path}不存在")
            print(f"捕捉到错误: {e}")

        # 按照索引找切片
        ct_slice = ct_array[start_slice, :, :]
        seg_slice = seg_array[start_slice, :, :]

        # 保存影像切片和分割标签切片
        ct_slice_path = os.path.join(preprocess_dir, 'ct', f'ct_{idx}')
        np.save(ct_slice_path, ct_slice)
        np.save(os.path.join(preprocess_dir, 'seg', f'seg_{idx}'), seg_slice)

        # 保存图片
        patient_id = subject_path.split('\\')[-1]
        # output_image_path = 'output_image.png'  # 请将此路径替换为你想要保存的文件路径
        plt.imsave(os.path.join(preprocess_dir, 'fig', f'ct_{patient_id}_{start_slice}.png'), ct_slice, cmap='gray')
        plt.imsave(os.path.join(preprocess_dir, 'fig', f'seg_{patient_id}_{start_slice}.png'), seg_slice, cmap='gray')

        idx += 1
