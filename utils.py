import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom
import torch
import torch
import torch.nn as nn
import torchmetrics
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torchvision.transforms.functional import InterpolationMode

from loss.Diceloss import dice_coefficient, DiceLoss


# 修改字体颜色
def prRed(skk):  return "\033[91m {}\033[00m" .format(skk)
def prGreen(skk): return "\033[92m {}\033[00m" .format(skk)


class Metrics():

    def __init__(self):

        # 实例化相关metrics的计算对象

        # self.dice = torchmetrics.Dice(average='samples').to('cuda')
        self.dice_corr = dice_coefficient
        self.dice_loss = DiceLoss()
        self.acc = torchmetrics.Accuracy(task='binary').to('cuda')
        # self.bceloss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.bceloss = torch.nn.BCELoss(reduction='mean')
        self.pred = []
        self.label = []
        self.batch_result = []
        self.batch_loss = 0
        # 收集每个epoch的loss绘图等
        self.epoch_loss = []
        self.epoch_dsc = []
        # 收集1次batch的loss
        self.batch_loss = []
        self.batch_dsc = []
        # 保存batch最新的平均loss
        self.batch_mean_loss = 0
        self.batch_mean_dsc = 0
        self.sig = nn.Sigmoid()

    def loss(self, pred, label, sigmoid=True):
        if sigmoid == True: pred = self.sig(pred)
        loss_value = self.bceloss(pred, label) + self.dice_loss(pred, label)
        # loss_value = self.dice_loss(pred, label)
        # loss_value = self.bceloss(pred, label)
        return loss_value


    def collect_batch(self, loss, pred, label, sigmoid=True):
        if sigmoid == True: pred = self.sig(pred)
        self.batch_loss.append(loss)

        batch_dsc = self.dice_corr(pred.detach(), label.detach())
        self.batch_dsc.append(batch_dsc.item())

        self.batch_mean_dsc = sum(self.batch_dsc) / len(self.batch_dsc)
        self.batch_mean_loss = sum(self.batch_loss) / len(self.batch_loss)

    def collect_epoch(self):
        self.epoch_loss.append(sum(self.batch_loss) / len(self.batch_loss))
        self.epoch_dsc.append(sum(self.batch_dsc) / len(self.batch_dsc))

    # def calc_batch_acc(self):
    #     batch_acc = self.calculate_accuracy(self.batch_result[0], self.batch_result[1])
    #     return batch_acc

    def save_result(self):
        pass

    def batch_reset(self):
        self.collect_epoch()
        # 重新归0
        self.batch_loss = []
        self.batch_dsc = []

    def draw(self, name, data_type, fig_type):
        if data_type == 'loss':
            data = self.epoch_loss
        if fig_type == 'line':
            self.draw_line(name, data)

    def draw_line(self, name, data):
        plt.plot(data, marker='*', linestyle='-', markersize=10, color='green', markerfacecolor='orange',
                 markeredgecolor='orange')
        # 添加标题和轴标签
        plt.title(' ')
        plt.xlabel('Epoch', fontsize=14, fontweight='bold')
        plt.ylabel('loss', fontsize=14, fontweight='bold')

        # 设置坐标轴刻度字体加大
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')

        # 显示图形
        # plt.grid(True)
        # 保存图像
        plt.savefig(f'results/{name}.png', dpi=600, bbox_inches='tight')
        # plt.show()
        # 清除窗口并准备新的绘图
        plt.clf()


def load_dicom(PATH):
    # 处理单个dicom切片, 使其对应模型的输入
    dicom_data = pydicom.dcmread(PATH)
    image_value = dicom_data.pixel_array
    return image_value


def normalize_dicom(image):
    # 对单个切片进行标准化
    # 归一化到 [0, 1]
    norm_image_data = (image - np.min(image)) / (np.max(image) - np.min(image))
    # 归一化到 [0, 255]
    norm_image_data = (norm_image_data * 255).astype(np.uint8)
    rgb_data = np.stack([norm_image_data] * 3, axis=-1)
    image = Image.fromarray(rgb_data, mode='RGB')
    return image


def dicom_totensor(image, size, label=None):
    # 转成tensor, 尺寸裁剪等

    if label:
        transform = transforms.Compose([
            myResize(size, size),
            myToTensor(),
            myNormalize(),
        ])
        image = transform(image)
    else:
        transform = transforms.Compose([
            transforms.Resize([size, size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        image = transform(image)
    return image


def numpy_to_nii(numpy_data, dicom_path, save_path):
    # 将分割后的结果保存为nii文件, 同时保存dicom的信息

    # 从 DICOM 文件读取原始图像
    dicom_image = sitk.ReadImage(dicom_path)
    # 从 DICOM 图像中提取头信息
    origin = dicom_image.GetOrigin()
    spacing = dicom_image.GetSpacing()
    direction = dicom_image.GetDirection()
    # # 在数据矩阵的最后增加一个维度，使其变为 (H, W, 1)
    # data_matrix_3d = np.expand_dims(numpy_data, axis=0)

    nii_file = sitk.GetImageFromArray(numpy_data)
    # anno_mat 为一个矩阵，其维度必须是按照(样本数*高度*宽度)排列
    # 设置头信息
    nii_file.SetOrigin(origin)
    nii_file.SetSpacing(spacing)
    nii_file.SetDirection(direction)
    sitk.WriteImage(nii_file, save_path)  # nii_path 为保存路径


# 对分割结果进行可视化
def vis_seg_pred_nolabel(image, pred, name="test.png"):

    image = image.cpu().numpy()
    pred = pred.cpu().numpy()

    # 处理image
    image = image / 2 + 0.5
    image = np.transpose(image, (1, 2, 0))

    # 创建一个图像，包含两个子图
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # 显示原始图像
    # axes[0].imshow(image, cmap='gray')
    axes[0].imshow(image)
    axes[0].set_title('Raw Image')
    axes[0].axis('off')

    # 将预测图像和分割标签保存为一张图像

    pred = (pred > 0.5)
    # 创建一个RGB图像，初始化为全黑
    pred = np.squeeze(pred)
    shape = pred.shape
    combined_image = np.zeros((shape[0], shape[1], 3), dtype=np.int32)
    # 为分割结果设置颜色 (例如，红色)
    combined_image[pred == 1] = [255, 0, 0]

    # 显示原始图像
    axes[1].imshow(combined_image)
    axes[1].set_title('Label')
    axes[1].axis('off')

    # 调整布局并显示图像
    plt.tight_layout()
    plt.show()
    save_path = os.path.join('SegResults', name)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


def save_nii_overlap_seg(numpy_data, dicom_path, save_path):
    # numpy_data 分割结果
    # dicom_path 原始影响路径
    # save_path 保存路径
    # 读取单个DICOM文件
    image = sitk.ReadImage(dicom_path)
    # 将图像数据转换为numpy数组 (确保是灰度图像)
    image_array = sitk.GetArrayFromImage(image)[0]  # 对于2D图像，GetArrayFromImage返回的是 (1, H, W) 的数组，需要提取第一个维度
    # 归一化图像数据到0到1范围
    # image_array_normalized = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    # 假设分割结果与图像相同大小，生成或读取分割结果
    # 将分割结果叠加到图像上
    # 将分割区域叠加到归一化的灰度图像上 (叠加的区域值为1)
    overlay_image = np.copy(image_array)
    overlay_image[numpy_data == 1] = np.max(image_array)  # 这里使用1表示病灶区域

    # 将叠加图像转换为SimpleITK格式
    overlay_sitk_image = sitk.GetImageFromArray(np.expand_dims(overlay_image, axis=0))

    # 复制原始图像的元数据（例如方向、间距、原点）到新的叠加图像
    overlay_sitk_image.CopyInformation(image)

    # 将叠加结果保存为NIfTI文件
    sitk.WriteImage(overlay_sitk_image, save_path)

class myToTensor:
    def __init__(self, dtype=torch.float32):
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
        return (torch.resize(image, [self.size_h, self.size_w], antialias=True),
                torch.resize(mask, [self.size_h, self.size_w], antialias=True, interpolation=InterpolationMode.NEAREST))

class myNormalize:
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    def __call__(self, data):
        img, msk = data
        img = self.normalize(img)
        return img, msk


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


def dice_coefficient(preds, targets, epsilon=1e-6):
    """
    计算 Dice 系数

    参数:
        preds (Tensor): 预测的分割图像张量，形状为 (N, H, W) 或 (N, C, H, W)
        targets (Tensor): 真实的分割图像张量，形状与 preds 相同
        epsilon (float): 防止除零错误的一个小常量

    返回:
        float: Dice 系数
    """
    # 确保张量是二值化的
    preds = (preds > 0.5)
    # targets = targets > 0.5
    num = preds.size(0)
    # 将预测和目标张量展平
    preds_flat = preds.view(num, -1)
    targets_flat = targets.view(num, -1)

    # 计算交集和并集
    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum()

    # 计算 Dice 系数
    dice = 2. * intersection / (union + epsilon)

    return dice