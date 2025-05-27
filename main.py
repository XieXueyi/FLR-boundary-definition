import numpy as np
import os.path
import platform
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from matplotlib.image import imsave
from tqdm import tqdm

from Datasets.Xiangya import get_dataset
from config import OptInit
from datasets import get_DataLoader
from models.ResUNet import ResUNet
from utils import *


def train():
    pass


def test():
    pass


class EarlyStop():
    def __init__(self, args):
        self.Maxepoch = args.MaxEpoch
        self.is_stop = False
        self.count = 0
        self.Min_loss = 999

    def collect_loss(self, loss):
        if loss < self.Min_loss:
            self.Min_loss = loss
            self.count = 0
        else:
            self.count += 1
            if self.count > self.Maxepoch:
                self.is_stop = True
        return self.is_stop


def get_model(args):
    if args.model == 'Unet':
        model = smp.Unet(
                encoder_name="resnet18",        # Choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                decoder_channels=[16, 32, 64, 128, 256],
                encoder_weights="imagenet",     # Use pre-trained weights from ImageNet
                in_channels=3,                  # Input channels (RGB images have 3 channels)
                classes=1,                      # Number of output classes (1 for binary segmentation)
            ).to(args.device)
    
    elif args.model == 'EGEUnet':
        from models.egeunet import EGEUNet
        model = EGEUNet(gt_ds=True).to(args.device)
    return model


def main(train_dataloder, val_dataloder,test_dataloder, args):
    # 定义模型和优化器
    model = get_model(args)

    earlystop = EarlyStop(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999), eps=1e-8)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

    # 创建指标收集器
    train_metric = Metrics()
    val_metric = Metrics()
    test_metric = Metrics()
    Min_loss = 999
    Best_epoch = 0
    for iter in range(args.epoch):
    ### Trian ###
        model.train()
        train_bar = tqdm(train_dataloder, desc=prGreen(f"train-epoch {iter + 1}"), colour='cyan')
        # batch_loss = []
        for batch_data, labels in train_bar:
            #将数据当到GPU上
            batch_data, labels = batch_data.to(args.device), labels.to(args.device)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            pred = model(batch_data)
            # 计算损失
            loss = train_metric.loss(pred, labels)
            # batch_loss.append(loss.item())
            train_metric.collect_batch(loss.item(), pred.detach(), labels.detach())
            # loss = dice_coefficient(pred, labels)
            # 反向传播计算梯度
            loss.backward()
            # 更新模型
            optimizer.step()

            # # 收集结果计算指标
            # train_metric.collect_batch(pred.cpu(), labels.cpu(), loss)
            # 进度条更新
            # train_bar.set_postfix({'loss': train_metric.dice.compute().item()})
            train_bar.set_postfix({'lr': lr_scheduler.get_last_lr()[0],
                                   'dsc': train_metric.batch_mean_dsc,
                                   'loss': train_metric.batch_mean_loss,
                                   })
            # train_bar.set_postfix({})
        # 收集每个epoch的loss
        train_metric.batch_reset()
        lr_scheduler.step()

        ### val ###

        model.eval()
        val_bar = tqdm(val_dataloder, desc=prGreen(f"val-epoch {iter + 1}"), colour='blue')
        # 不计算梯度，节省内存
        with torch.no_grad():
            for batch_data, labels in val_bar:
                batch_data, labels = batch_data.to(args.device), labels.to(args.device)
                # labels = labels.to(dtype=torch.float32)
                # 前向传播
                pred = model(batch_data)
                # 计算损失
                # loss = val_metric.dice(pred, labels)
                loss = val_metric.loss(pred, labels)
                val_metric.collect_batch(loss.item(), pred.detach(), labels.detach())

                val_bar.set_postfix({'lr': lr_scheduler.get_last_lr()[0],
                                     'earlystop': earlystop.count,
                                     'dsc': val_metric.batch_mean_dsc,
                                     'loss': val_metric.batch_mean_loss})

        # 保存val集上最好的模型
        if val_metric.batch_mean_loss < Min_loss:
            Best_epoch = iter + 1
            Min_loss = val_metric.batch_mean_loss
            torch.save(model.state_dict(), f'checkpoint/model_{args.size}.pth')
        is_stop = earlystop.collect_loss(val_metric.batch_mean_loss)
        if is_stop:
            break
        val_metric.batch_reset()

    ### test ###

    # 加载验证机最好模型
    model.load_state_dict(torch.load(f'checkpoint/model_{args.size}.pth'))
    model.eval()
    test_bar = tqdm(test_dataloder, desc=prGreen(f"Start Test:"), colour='red')
    # 不计算梯度，节省内存

    with torch.no_grad():
        for batch_data, labels in test_bar:
            batch_data, labels = batch_data.to(args.device), labels.to(args.device)
            # labels = labels.to(dtype=torch.float32)
            # 前向传播
            pred = model(batch_data)
            # 计算损失
            # loss = val_metric.dice(pred, labels)
            # 计算dice相似系数
            loss = test_metric.loss(pred, labels)
            test_metric.collect_batch(loss.item(), pred.detach(), labels.detach())
            # val_bar.set_postfix({'loss': val_metric.dice.compute().item()})
            # test_metric.batch_mean_loss = sum(test_metric.batch_loss) / len(test_metric.batch_loss)
            test_bar.set_postfix({'loss': test_metric.batch_mean_loss})
    print("Result in Test:")
    print(f"Best_epoch: {Best_epoch}, loss: {test_metric.batch_mean_loss}, dsc: {test_metric.batch_mean_dsc}")

    # 结果可视化
    train_metric.draw(name='train_bceloss', data_type='loss', fig_type='line')
    val_metric.draw(name='val_bceloss', data_type='loss', fig_type='line')
    from visualization.draw_line import draw_2_line
    draw_2_line(train_metric.epoch_loss, val_metric.epoch_loss, name='train-val-loss')
    draw_2_line(train_metric.epoch_dsc, val_metric.epoch_dsc, name='train-val-dsc')


if __name__ == "__main__":
    #  参数初始化
    opt = OptInit()
    opt.initialize()
    system_name = platform.system()

    ROOT = "../data/"

    ROOT_PATH = os.path.join(ROOT, opt.args.dataset)
    train_dataloder, val_loader, test_dataloder = get_dataset(ROOT_PATH, opt.args)
    print(opt.args)
    main(train_dataloder, val_loader, test_dataloder, opt.args)
    print("finish")