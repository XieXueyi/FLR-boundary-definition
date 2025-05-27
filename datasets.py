import numpy as np
import os.path
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from config import OptInit


class LiverDataSet(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X = np.load(self.df['image_path'][index]).astype(np.float32)
        y = np.load(self.df['label'][index])
        y = torch.tensor(y).unsqueeze(0)

        if self.transform:
            X = self.transform(X)
        return X, y


def get_DataLoader(arg, ROOT_PATH):

    transform = transforms.Compose([
        transforms.ToPILImage(),  # 将 NumPy 数组或张量转换为 PIL 图像
        transforms.ToTensor(),  # 将 PIL 图像转换为张量，并将值归一化到 [0, 1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化图像，假设图像值范围在 [0, 1]
    ])

    ct_file = []
    seg_file = []

    ct_path = os.path.join(ROOT_PATH, 'ct')
    seg_path = os.path.join(ROOT_PATH, 'seg')
    for ct_name, seg_name in zip(os.listdir(ct_path), os.listdir(seg_path)):
        ct_name = os.path.join(ct_path, ct_name)
        seg_name = os.path.join(seg_path, seg_name)
        ct_file.append(ct_name)
        seg_file.append(seg_name)

    # 创建DataFrame
    df = pd.DataFrame({
        'image_path': ct_file,
        'label': seg_file
    })
    dataset = LiverDataSet(df, transform)

    # 定义训练集、验证集和测试集的划分比例
    train_ratio = 0.8
    test_ratio = 0.2

    train_dataset, test_dataset = random_split(dataset, [train_ratio, test_ratio])
    train_loader = DataLoader(train_dataset, batch_size=arg.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=arg.batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


if __name__ == "__main__":
    opt = OptInit()
    opt.initialize()
    get_DataLoader(' ', '..\data\XiangYa_process')
