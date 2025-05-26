import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QFileDialog, QTextEdit
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import segmentation_models_pytorch as smp
import torch
from torch import nn
import utils
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import os

class ModelPredictionApp(QWidget):
    def __init__(self):
        super().__init__()

        # 创建主布局
        self.layout = QVBoxLayout()

        # 第1行: 模型创建
        self.model_layout = QHBoxLayout()
        self.model_label = QLabel("选择模型:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["ResNet18_Unet", "ResNet34_Unet", "ResNet50_Unet"])  # Example models
        self.create_button = QPushButton("创建模型")
        self.model_layout.addWidget(self.model_label)
        self.model_layout.addWidget(self.model_combo)
        self.model_layout.addWidget(self.create_button)
        self.layout.addLayout(self.model_layout)

        # 第2行: 加载模型权重
        self.weight_layout = QHBoxLayout()
        self.weight_label = QLabel("加载模型权重:")
        self.weight_button = QPushButton("选择文件")
        self.weight_layout.addWidget(self.weight_label)
        self.weight_layout.addWidget(self.weight_button)
        self.layout.addLayout(self.weight_layout)

        # 第3行: 选择测试数据
        self.data_layout = QHBoxLayout()
        self.data_label = QLabel("选择测试数据:")
        self.data_button = QPushButton("选择文件")
        self.data_layout.addWidget(self.data_label)
        self.data_layout.addWidget(self.data_button)
        self.layout.addLayout(self.data_layout)

        # 第4行: 模型预测
        self.prediction_layout = QHBoxLayout()
        self.prediction_label = QLabel("模型预测:")
        self.prediction_button = QPushButton("开始")
        self.prediction_layout.addWidget(self.prediction_label)
        self.prediction_layout.addWidget(self.prediction_button)
        self.layout.addLayout(self.prediction_layout)

        # 第5行: 结果可视化
        self.visualize_layout = QHBoxLayout()
        self.visualize_label = QLabel("结果可视化:")
        self.visualize_button = QPushButton("显示结果")
        self.visualize_layout.addWidget(self.visualize_label)
        self.visualize_layout.addWidget(self.visualize_button)
        self.layout.addLayout(self.visualize_layout)

        # 第6行: 保存标签
        self.save_layout = QHBoxLayout()
        self.save_label = QLabel("保存结果:")
        self.save_button = QPushButton("分割标签")
        self.save_button_withRawData = QPushButton("原始影像和标签")
        self.save_layout.addWidget(self.save_label)
        self.save_layout.addWidget(self.save_button)
        self.save_layout.addWidget(self.save_button_withRawData)
        self.layout.addLayout(self.save_layout)

        # 第5行: 输出框
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.layout.addWidget(self.output_text)

        # 设置主窗口属性
        self.setLayout(self.layout)
        self.setWindowTitle("模型预测界面")
        self.setGeometry(300, 300, 600, 400)

        # 连接信号和槽
        self.create_button.clicked.connect(self.create_model)
        self.weight_button.clicked.connect(self.load_weights)
        self.data_button.clicked.connect(self.load_data)
        self.prediction_button.clicked.connect(self.prediction)
        self.visualize_button.clicked.connect(self.visualize_results)
        self.save_button_withRawData.clicked.connect(self.save_prediction_withRawData)
        self.save_button.clicked.connect(self.save_prediction)


        # 模型参数设置等
        # 初始化模型属性
        self.model = None
        self.test_data = None
        self.result = None
        self.input_filepath = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def create_model(self):
        model_name = self.model_combo.currentText()
        if model_name == 'ResNet18_Unet': encoder_name = "resnet18"
        elif model_name == 'ResNet34_Unet': encoder_name = "resnet34"
        elif model_name == 'ResNet50_Unet': encoder_name = "resnet50"

        # 创建模型
        self.model = smp.Unet(
            encoder_name = encoder_name,
            decoder_channels=[16, 32, 64, 128, 256],
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        ).to(self.device)

        self.output_text.append(f"创建模型: {model_name}")

    def load_weights(self):
        if self.model is None:
            self.output_text.append("请先创建模型！")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型权重文件", "", "Model Files (*.h5 *.pt *.pth)")
        # torch读取模型权重
        model_weight = torch.load(file_path)
        # # 加载模型
        self.model.load_state_dict(model_weight)

        if file_path:
            self.output_text.append(f"加载模型权重: {file_path}")

    def load_data(self):
        if self.model is None:
            self.output_text.append("请先创建模型！")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "选择测试数据文件", "", "Data Files (*.dcm)")
        if file_path:
            self.output_text.append(f"选择测试数据: {file_path}")
            self.input_filepath = file_path
        dcm_data = utils.load_dicom(file_path)
        self.output_text.append(f"数据读取成功: {file_path}")
        self.output_text.append(f"数据处理中...")

        dcm_data = utils.normalize_dicom(dcm_data)
        dcm_data = utils.dicom_totensor(dcm_data, size=256)
        self.test_data = dcm_data.to(self.device)
        self.output_text.append(f"数据处理完成...")

    def prediction(self):
        if self.test_data is None:
            self.output_text.append("请先读取测试数据！")
            return
        self.output_text.append("模型预测中...")
        self.result = self.model(self.test_data.unsqueeze(0))
        self.output_text.append("模型预测完成...")

    def visualize_results(self):
        if self.result is None:
            self.output_text.append("请先生成分割结果！")
            return

        sig = nn.Sigmoid()
        self.result = sig(self.result)
        self.output_text.append("结果可视化中...")
        utils.vis_seg_pred_nolabel(self.test_data, self.result.detach())

    def save_prediction(self):
        if self.result is None:
            self.output_text.append("请先生成分割结果！")
            return

        # 设置默认文件名
        base_name = os.path.basename(self.input_filepath)
        file_name_without_ext = os.path.splitext(base_name)[0]
        default_filename = f"{file_name_without_ext}_prediction.nii"


        resize_512 = transforms.Resize([512, 512], interpolation=InterpolationMode.NEAREST)
        result = torch.where(self.result > 0.5, 1, 0)
        result_512 = resize_512(result)


        # 弹出保存文件对话框
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(self,
                                                   "保存文件",
                                                   default_filename,
                                                   "所有文件 (*);;影像文件 (*.nii)",
                                                   options=options)
        if file_path:
            utils.numpy_to_nii(result_512.detach().cpu().numpy().squeeze(axis=0), self.input_filepath, save_path=file_path)
        self.output_text.append(f"文件已保存至: {file_path}")

    def save_prediction_withRawData(self):
        if self.result is None:
            self.output_text.append("请先生成分割结果！")
            return

        # 设置默认文件名
        base_name = os.path.basename(self.input_filepath)
        file_name_without_ext = os.path.splitext(base_name)[0]
        default_filename = f"{file_name_without_ext}_prediction_rawdata.nii"
        #
        #
        resize_512 = transforms.Resize([512, 512], interpolation=InterpolationMode.NEAREST)
        result = torch.where(self.result > 0.5, 1, 0)
        result_512 = resize_512(result)

        # 弹出保存文件对话框
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(self,
                                                   "保存文件",
                                                   default_filename,
                                                   "所有文件 (*);;影像文件 (*.nii)",
                                                   options=options)

        if file_path:
            # utils.numpy_to_nii(result_512.detach().cpu().numpy().squeeze(axis=0), self.input_filepath, save_path=file_path)
            utils.save_nii_overlap_seg(result_512.detach().cpu().numpy().squeeze(), self.input_filepath, save_path=file_path)
        self.output_text.append(f"文件已保存至: {file_path}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 设置全局字体
    font = QFont("Arial", 12)
    app.setFont(font)
    window = ModelPredictionApp()
    window.show()
    sys.exit(app.exec_())
