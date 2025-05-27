import matplotlib as plt
import numpy as np
import os.path
import pandas as pd
import random
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import tqdm
from PIL import Image
from matplotlib.image import imsave
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms.functional import InterpolationMode

from config import OptInit


def _init_fn():
    np.random.seed(0)


class CustomDataset(Dataset):
    def __init__(self, df, data_transform=None):
        self.df = df
        self.data_transform = data_transform

        # 扫描数据到内存
        self.image_list = [0 for _ in range(len(self.df))]
        self.seg_list = [0 for _ in range(len(self.df))]
        self.load_memory = np.random.choice([False], size=len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.load_memory[idx] == False:
            img_path = self.df['image'].iloc[idx]
            label_path = self.df['label'].iloc[idx]
            image = np.load(img_path)
            label = np.load(label_path)
            self.image_list[idx] = image
            self.seg_list[idx] = label

            self.load_memory[idx] = True
        else:
            image = self.image_list[idx]
            label = self.seg_list[idx]
        # 归一化到 [0, 1]
        norm_image_data = (image - np.min(image)) / (np.max(image) - np.min(image))
        # 归一化到 [0, 255]
        norm_image_data = (norm_image_data * 255).astype(np.uint8)
        rgb_data = np.stack([norm_image_data] * 3, axis=-1)
        image = Image.fromarray(rgb_data, mode='RGB')

        label = label.astype(np.uint8) * 255 # 归一化到 [0, 255]
        label = Image.fromarray(label, mode='L')
        if self.data_transform:
            image, label = self.data_transform((image, label))
        return image, label


def get_dataset(ROOT_PATH, args):

    ct_path = os.path.join(ROOT_PATH, 'ct')
    seg_path = os.path.join(ROOT_PATH, 'seg')

    ct_list = []
    seg_list = []

    for ct_name, seg_name in zip(sorted(os.listdir(ct_path)) , sorted(os.listdir(seg_path))):
        ct_list.append(os.path.join(ct_path, ct_name))
        seg_list.append(os.path.join(seg_path, seg_name))

    df = pd.DataFrame({'image':ct_list, 'label': seg_list})
    # 打乱 DataFrame 的索引
    indices = np.arange(len(df))
    np.random.shuffle(indices)

    # 划分比例
    train_size = int(0.7 * len(df))  # 60% 用于训练
    val_size = int(0.1 * len(df))  # 20% 用于验证
    test_size = len(df) - train_size - val_size  # 剩下的 20% 用于测试

    # 划分索引
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # 创建训练集、验证集和测试集
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    print("训练集:\n", train_df)
    print("验证集:\n", val_df)
    print("测试集:\n", test_df)

    train_data_transform = transforms.Compose([
        myResize(args.size, args.size),
        myToTensor(),
        myNormalize(),
    ])

    test_data_transform = transforms.Compose([
        myResize(args.size, args.size),
        myToTensor(),
        myNormalize(),
    ])

    train_dataset = CustomDataset(df=train_df, data_transform=train_data_transform)
    val_dataset = CustomDataset(df=val_df, data_transform=test_data_transform)
    test_dataset = CustomDataset(df=test_df, data_transform=test_data_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, worker_init_fn=_init_fn())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader, test_loader


class myToTensor:
    def __init__(self, dtype = torch.float32):
        self.dtype = dtype
        self.totensor = transforms.ToTensor()
    def __call__(self, data):
        image, mask = data
        return self.totensor(image), self.totensor(mask)


class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        image, mask = data
        return (TF.resize(image, [self.size_h, self.size_w], antialias=True),
                TF.resize(mask, [self.size_h, self.size_w], antialias=True, interpolation=InterpolationMode.NEAREST))


class myNormalize:
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __call__(self, data):
        img, msk = data
        img = self.normalize(img)
        return img, msk


if __name__ == "__main__":
    opt = OptInit()
    opt.initialize()
    ROOT_PATH = '..\data\XiangYa'

    train_loader, val_loader, test_loader = get_dataset(ROOT_PATH, opt.args)

    for images, labels in tqdm.tqdm(val_loader):
        first_img = images[0].numpy()
        first_lab = labels[0].numpy()
        first_img = first_img / 2 + 0.5
        imsave('image.png', np.transpose(first_img, (1, 2, 0)))
        imsave('label.png', first_lab[0], cmap='gray')
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels shape: {labels.shape}")
