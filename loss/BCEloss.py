import torch
import torch.nn as nn
import torch.optim as optim

# 简单的分割模型
class SimpleSegmentationModel(nn.Module):
    def __init__(self):
        super(SimpleSegmentationModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        # x = self.sigmoid(x)
        return x

# 初始化模型、损失函数和优化器
model = SimpleSegmentationModel()
# BCEcriterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()
criterion_sum = nn.BCEWithLogitsLoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=0.001)
sig = nn.Sigmoid()
# 示例输入和标签（单通道的2x2图像）
input_image = torch.randn(2, 1, 2, 2)  # 单张图片 (batch_size=1, channels=1, height=2, width=2)
target_mask = torch.tensor([[[[0., 1.], [1., 0.]]],
                            [[[0., 1.], [1., 0.]]]
                            ])  # 对应的标签

# 训练步骤
model.train()
optimizer.zero_grad()
output = model(input_image)

# 计算损失
loss = criterion(output, target_mask)
print('Loss:', loss.item())
loss = criterion_sum(output.reshape(2,-1), target_mask.reshape(2,-1))
print('Loss:', loss.item())

# 反向传播和优化
loss.backward()
optimizer.step()
