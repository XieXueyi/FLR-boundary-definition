import torch
from torchmetrics import Dice
import torch.nn as nn
import numpy as np

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


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


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss


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




if __name__ == "__main__":

    # 设置阈值
    threshold = 0.5
    # t_dice = Dice(average='samples')

    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)

    # 随机生成3张2D预测结果，尺寸为4x4，值在0到1之间
    predictions = torch.rand(3, 1, 4, 4)
    binary_predictions = (predictions > threshold).float()
    # print(binary_predictions)
    # 随机生成3张2D标签，尺寸为4x4，值为0或1
    labels = torch.randint(0, 2, (3, 1, 4, 4))

    # print("Predictions:\n", predictions)
    # print("Labels:\n", labels)

    # out_t_dice = t_dice(predictions, labels)
    # out_t_dice_threshold = t_dice(binary_predictions, labels)
    # print(out_t_dice)
    # print(out_t_dice_threshold)

    # dl = diceLoss()
    # dice = dl(predictions, labels)
    # print(dice)
    dsc_1 = dice_coefficient(predictions[0], labels[0])
    dsc_2 = dice_coefficient(predictions[1], labels[1])
    dsc_3 = dice_coefficient(predictions[2], labels[2])
    dsc = dice_coefficient(predictions, labels)
    # dsc_2 = dice_loss(predictions, labels)
    print(dsc_1)
    print(dsc_2)
    print(dsc_3)
    print((dsc_3+dsc_2+dsc_1) / 3)
    print(dsc)
    # print(dsc_2)