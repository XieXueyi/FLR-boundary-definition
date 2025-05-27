import matplotlib.pyplot as plt



def draw_2_line(data1, data2, name):
    # 创建图形
    plt.figure(figsize=(10, 5))

    # 绘制训练集的loss曲线
    plt.plot(data1, label='Train', marker='o')

    # 绘制验证集的loss曲线
    plt.plot(data2, label='Val', marker='*')

    # 添加标题和标签
    plt.title(f'{name}')


    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Metric', fontsize=14, fontweight='bold')

    # 设置坐标轴刻度字体加大
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    # 显示图例
    plt.legend()
    plt.savefig(f'results/{name}.png', dpi=600, bbox_inches='tight')
    # 显示图形
    # plt.grid(True)
    # plt.show()
    # 清除窗口并准备新的绘图
    plt.clf()
# if __name__ == "__main__":
    # 假设数据如下，替换为你的数据
    # epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # train_loss = [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25]
    # val_loss = [1.0, 0.9, 0.85, 0.8, 0.75, 0.72, 0.7, 0.68, 0.65, 0.63]
    # draw_2_line(train_loss, val_loss, name='test')