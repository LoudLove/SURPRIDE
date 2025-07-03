import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import ViTImageProcessor, ResNetForImageClassification, AutoModelForImageClassification
import torch
import torch.nn as nn
from transformers import ConvNextImageProcessor, ConvNextForImageClassification
from transformers import ViTForImageClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import MSELoss
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, kendalltau
import random
import matplotlib.pyplot as plt
from transformers import pipeline, set_seed
import torch.nn.functional as F
import time


def set_seed_self(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


# 计算 PLCC, SRCC, KLCC
def compute_metrics(predictions, labels):
    # Pearson Linear Correlation Coefficient (PLCC)
    plcc, _ = pearsonr(predictions, labels)

    # Spearman Rank Correlation Coefficient (SRCC)
    srcc, _ = spearmanr(predictions, labels)

    # Kendall Rank Correlation Coefficient (KLCC)
    klcc, _ = kendalltau(predictions, labels)

    return plcc, srcc, klcc


class IQAdataset(Dataset):
    def __init__(self, metadata, original_path, SR_paths,
                 target_size=384, patch_size=16):
        self.metadata = metadata
        self.original_path = original_path
        self.SR_paths = SR_paths
        self.target_size = target_size
        self.patch_size = patch_size
        self.num_patches = target_size // patch_size  # 计算每行/列的patch数量
        self.data_set = self.metadata.iloc[0]['set']

        # 引入缓存字典
        self.cache = {}

        # 定义 transform 方法，用于图像预处理
        self.transform = transforms.Compose([
            # 转换为 Tensor
            transforms.ToTensor(),
            # 标准化，mean 和 std 按照 [0.5, 0.5, 0.5]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 获取图像信息
        image_name = self.metadata.iloc[idx]['image_name']
        mos_score = self.metadata.iloc[idx]['quality_mos']

        # 如果该图像已被缓存，直接从缓存中获取
        if image_name in self.cache:
            return self.cache[image_name]  # 返回缓存的结果

        # 构建图像路径
        original_img_path = os.path.join(self.original_path, self.data_set, image_name)
        image_name_png = os.path.splitext(image_name)[0] + '.png'  # 确保文件名是 .png 扩展名
        SRx4_img_path = os.path.join(self.SR_paths['SwinFIR_SRx4'], self.data_set, image_name_png)

        # 加载图像
        original_img = Image.open(original_img_path).convert("RGB") #【3840，不等】
        new_size = (2560, 1440)
        original_img_for_splice = original_img.resize(new_size, Image.LANCZOS)
        SRx4_img = Image.open(SRx4_img_path).convert("RGB") #【2560, 1440】

        # 将图像转换为 NumPy 数组
        original_img_np = np.array(original_img_for_splice)
        SRx4_img_np = np.array(SRx4_img)

        # 获取随机patches并拼接
        patch_positions = self._get_fixed_patches_pos(np.array(original_img_np))  # 获取固定的切分位置

        # 根据获取的 patch 位置切分图像并拼接
        original_img_splice = self.generate_img_from_patches(original_img_np, patch_positions)
        SRx4_img_splice = self.generate_img_from_patches(SRx4_img_np, patch_positions)

        # 使用偏移量重新切分图像
        offset_x = 53
        offset_y = 30

        patch_positions2 = self._get_fixed_patches_pos(np.array(original_img_np), offset_x, offset_y)
        original_img_splice2 = self.generate_img_from_patches(original_img_np, patch_positions2)
        SRx4_img_splice2 = self.generate_img_from_patches(SRx4_img_np, patch_positions2)

        # 将图像转换为模型输入
        original_img_splice_tensor = self.transform(original_img_splice)
        SRx4_img_splice_tensor = self.transform(SRx4_img_splice)
        original_img_splice_tensor2 = self.transform(original_img_splice2)
        SRx4_img_splice_tensor2 = self.transform(SRx4_img_splice2)

        # 构造返回数据
        data = (
            original_img_splice_tensor, SRx4_img_splice_tensor,
            original_img_splice_tensor2, SRx4_img_splice_tensor2,
            torch.tensor(mos_score, dtype=torch.float32)
        )

        # 将当前数据缓存起来
        self.cache[image_name] = data

        # 返回原图、重采样图、拼接图像以及MOS分数
        return data

    def _get_fixed_patches_pos(self, img, offset_x=0, offset_y=0):
        """
        获取图像的固定位置 patch 切分。
        :param img: 输入图像（NumPy数组）。
        :param offset_x: 横向偏移量。
        :param offset_y: 纵向偏移量。
        :return: 均匀切分的 patch 位置。
        """
        patch_positions = []

        # 获取图像的宽度和高度
        height, width, _ = img.shape

        # 计算每行和每列的 patches 数量
        num_patches = self.target_size // self.patch_size

        # 计算步长
        row_step = (height - self.patch_size) // num_patches
        col_step = (width - self.patch_size) // num_patches

        # 均匀切分图像的 patch 位置
        for i in range(num_patches):
            for j in range(num_patches):
                top = i * row_step + offset_y
                left = j * col_step + offset_x
                patch_positions.append((top, left))  # 均匀切分

        return patch_positions

    def generate_img_from_patches(self, img, patch_positions):
        """
        根据提供的 patch 位置返回拼接后的图像。
        :param img: 输入图像（NumPy 数组）。
        :param patch_positions: 每个 patch 的位置和类型列表。
        :param patch_size: 每个 patch 的大小（默认32）。
        :return: 拼接后的图像。
        """
        patches = []

        # 根据 patch_positions 切分图像
        for (top, left) in patch_positions:
            patch = img[top:top + self.patch_size, left:left + self.patch_size]
            patches.append(patch)

        # 计算每行需要拼接的 patch 数量
        num_patches = int(np.sqrt(len(patches)))  # 假设是一个正方形网格

        # 直接按照每行拼接每个 patch
        rows = []
        for i in range(num_patches):
            # 将每一行的 patch 按列拼接
            row_patches = [patches[i * num_patches + j] for j in range(num_patches)]

            # 在列方向拼接（即第二个维度）
            row = np.concatenate(row_patches, axis=1)  # 在列方向上拼接
            rows.append(row)

        # 按行拼接所有行（即在第一个维度）
        img_np = np.concatenate(rows, axis=0)  # 在行方向拼接

        # 返回拼接后的图像张量
        return img_np


class MultiViTRegressor(nn.Module):
    def __init__(self, model_name="facebook/convnext-base-384-22k-1k", feat_weight=0.05):
        super(MultiViTRegressor, self).__init__()
        self.feat_weight = feat_weight

        # 初始化6个独立的 ViT 模型
        self.model1 = ConvNextForImageClassification.from_pretrained(model_name)
        self.model2 = ConvNextForImageClassification.from_pretrained(model_name)
        # self.model3 = ConvNextForImageClassification.from_pretrained(model_name)
        # self.model4 = ConvNextForImageClassification.from_pretrained(model_name)
        # print(self.model1)

        # 替换分类头为 Identity
        self.model1.classifier = nn.Identity()
        self.model2.classifier = nn.Identity()
        # self.model3.classifier = nn.Identity()
        # self.model4.classifier = nn.Identity()

        # 定义回归头：输入为拼接的 6x768 维特征
        self.regressor = nn.Sequential(
            nn.Linear(1024 * 2, 512),
            nn.Linear(512, 1)  # 回归输出为 1
        )

    def forward(self, img1, img2):
        # 通过每个 ViT 模型提取特征
        feat1 = self.model1(img1).logits
        feat2 = self.model2(img2).logits
        # feat3 = self.model3(img3).logits
        # feat4 = self.model3(img4).logits

        # 对 feat2 进行加权，使得它只占最终特征融合的 20%
        feat2_weighted = feat2 * self.feat_weight
        feat1_weighted = feat1 * (1 - self.feat_weight)

        # 拼接加权后的特征
        combined_features = torch.cat([feat1_weighted, feat2_weighted], dim=-1)

        cosine_similarity = F.cosine_similarity(feat1_weighted, feat2_weighted, dim=-1).mean()
        # F.cosine_similarity 的结果会返回一个介于 [-1, 1] 之间的值

        # 拼接6个特征
        # combined_features = torch.cat([feat1, feat2], dim=-1)

        # 通过回归头输出
        output = self.regressor(combined_features)
        return output, cosine_similarity



def search_hyperparameters():
    set_seed(42)
    set_seed_self()

    # 设置设备：如果有 GPU 则使用 GPU，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 读取metadata文件
    metadata_path = r'J:\DATASET\uhd-iqa\UHD-IQA-database\uhd-iqa-metadata.csv'
    metadata = pd.read_csv(metadata_path)

    # 根据 'set' 列筛选训练、验证、测试数据
    train_metadata = metadata[metadata['set'] == 'training']
    val_metadata = metadata[metadata['set'] == 'validation']
    test_metadata = metadata[metadata['set'] == 'test']

    # 超分图像路径
    SR_paths = {
        'SwinFIR_SRx2': r'J:\DATASET\SwinFIRx2_uhdiqa',
        'SwinFIR_SRx3': r'J:\DATASET\SwinFIRx3_uhdiqa',
        'SwinFIR_SRx4': r'J:\DATASET\SwinFIRx4_uhdiqa',
        'HAT_SRx2': r'J:\DATASET\uhdiqa_HAT_2x',
        'HAT_SRx3': r'J:\DATASET\uhdiqa_HAT_3x',
        'HAT_SRx4': r'J:\DATASET\uhdiqa_HAT_4x',
        'DRCT_SRx4': r'J:\DATASET\DRCT_uhdiqa',
    }

    # 初始化数据集
    train_dataset = IQAdataset(
        metadata=train_metadata,
        original_path=r'J:\DATASET\uhd-iqa\UHD-IQA-database',
        SR_paths=SR_paths
    )
    val_dataset = IQAdataset(
        metadata=val_metadata,
        original_path=r'J:\DATASET\uhd-iqa\UHD-IQA-database',
        SR_paths=SR_paths
    )
    test_dataset = IQAdataset(
        metadata=test_metadata,
        original_path=r'J:\DATASET\uhd-iqa\UHD-IQA-database',
        SR_paths=SR_paths
    )

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # 初始化模型，传递feat_weight
    model = MultiViTRegressor(model_name="facebook/convnext-base-384-22k-1k", feat_weight=0.3).to(
        device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = MSELoss()
    lambda_1 = 0.9

    # 训练循环
    best_val_loss = float('inf')  # 初始化一个非常大的验证损失值
    best_model_state = None  # 用来保存最佳模型的状态字典

    for epoch in range(15):
        model.train()
        total_loss = 0
        train_predictions = []
        train_targets = []

        with tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training", unit="batch") as pbar:
            for a, b, c, d, mos_score in pbar:
                # 将数据迁移到设备
                a, b, c, d, mos_score = a.to(device), b.to(device), c.to(device), d.to(device), mos_score.to(device)

                optimizer.zero_grad()

                # 前向传播
                output1, cos_simi = model(a, b)
                output2, cos_simi2 = model(c, d)  # for second input pair

                # 原始的 MSE loss
                mse_loss = criterion(output1.squeeze(), mos_score)
                mse_loss2 = criterion(output2.squeeze(), mos_score)

                # 总损失 = MSE损失 + 余弦相似度损失
                total_loss_per_batch1 = mse_loss + lambda_1 * (1 - torch.abs(cos_simi))   # 期待着两个分支之间强正相关或负相关
                total_loss_per_batch2 = mse_loss2 + lambda_1 * (1 - torch.abs(cos_simi2))

                # 计算总损失（对两个部分的损失求平均）
                total_loss_per_batch = (total_loss_per_batch1 + total_loss_per_batch2) / 2

                # 反向传播
                total_loss_per_batch.backward()
                optimizer.step()

                total_loss += total_loss_per_batch.item()
                # 计算平均的预测输出
                avg_output = (output1 + output2) / 2

                # 保存训练阶段的预测和真实值
                train_predictions.extend([o.item() for o in avg_output.detach().cpu().numpy()])
                train_targets.extend([m.item() for m in mos_score.detach().cpu().numpy()])

                # 更新进度条显示当前损失
                pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        # 计算 PLCC, SRCC, KLCC
        train_plcc, train_srcc, train_klcc = compute_metrics(train_predictions, train_targets)
        print(f"Epoch {epoch + 1}, Train Loss: {total_loss / len(train_loader)}")
        print(f"Train PLCC: {train_plcc}, Train SRCC: {train_srcc}, Train KLCC: {train_klcc}")

        # 验证循环
        model.eval()  # 验证模式
        val_loss = 0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch + 1} - Validation", unit="batch") as pbar:
                for a, b, c, d, mos_score in pbar:
                    a, b, c, d, mos_score = a.to(device), b.to(device), c.to(device), d.to(device), mos_score.to(device)

                    # 前向传播
                    output1, cos_simi1 = model(a, b)
                    output2, cos_simi2 = model(c, d)

                    # 计算损失
                    mse_loss1 = criterion(output1.squeeze(), mos_score)
                    mse_loss2 = criterion(output2.squeeze(), mos_score)
                    cos_simi_loss1 = 1 - torch.abs(cos_simi1)
                    cos_simi_loss2 = 1 - torch.abs(cos_simi2)

                    # 总损失 = MSE损失 + 余弦相似度损失
                    total_loss_per_batch1 = mse_loss1 + lambda_1 * cos_simi_loss1
                    total_loss_per_batch2 = mse_loss2 + lambda_1 * cos_simi_loss2

                    # 计算总损失（对两个部分的损失求平均）
                    total_loss_per_batch = (total_loss_per_batch1 + total_loss_per_batch2) / 2

                    val_loss += total_loss_per_batch.item()
                    # 计算平均的预测输出
                    avg_output = (output1 + output2) / 2

                    # 保存验证阶段的预测和真实值
                    val_predictions.extend([o.item() for o in avg_output.detach().cpu().numpy()])
                    val_targets.extend([m.item() for m in mos_score.detach().cpu().numpy()])

                    # 更新进度条显示当前损失
                    pbar.set_postfix(loss=val_loss / (pbar.n + 1))

        # 计算 PLCC, SRCC, KLCC
        val_plcc, val_srcc, val_klcc = compute_metrics(val_predictions, val_targets)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader)}")
        print(f"Validation PLCC: {val_plcc}, Validation SRCC: {val_srcc}, Validation KLCC: {val_klcc}")

        # 如果当前模型在验证集上的损失更小，保存当前模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # 保存模型的参数

    # 保存最优的模型
    if best_model_state is not None:
        torch.save(best_model_state, 'UHDIQA_exp_lambda_alpha_10.pth')
        print("Best model saved based on validation performance ")

    # 在测试前，加载保存的最佳模型
    if best_model_state is not None:
        model.load_state_dict(torch.load('UHDIQA_exp_lambda_alpha_10.pth'))
        print("Best model loaded for testing ")

    # 测试阶段计算
    model.eval()  # 测试模式
    test_loss = 0
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        with tqdm(test_loader, desc="Testing", unit="batch") as pbar:
            for a, b, c, d, mos_score in pbar:
                a, b, c, d, mos_score = a.to(device), b.to(device), c.to(device), d.to(device), mos_score.to(device)

                # 前向传播
                output1, cos_simi1 = model(a, b)
                output2, cos_simi2 = model(c, d)

                # 计算损失
                mse_loss1 = criterion(output1.squeeze(), mos_score)
                mse_loss2 = criterion(output2.squeeze(), mos_score)
                cos_simi_loss1 = 1 - torch.abs(cos_simi1)
                cos_simi_loss2 = 1 - torch.abs(cos_simi2)

                # 总损失 = MSE损失 + 余弦相似度损失
                total_loss_per_batch1 = mse_loss1 + lambda_1 * cos_simi_loss1  # 期待着两个分支之间正相关或负相关
                total_loss_per_batch2 = mse_loss2 + lambda_1 * cos_simi_loss2

                # 计算总损失（对两个部分的损失求平均）
                total_loss_per_batch = (total_loss_per_batch1 + total_loss_per_batch2) / 2

                test_loss += total_loss_per_batch.item()

                # 计算平均的预测输出
                avg_output = (output1 + output2) / 2

                # 保存测试阶段的预测和真实值
                test_predictions.extend([o.item() for o in avg_output.detach().cpu().numpy()])
                test_targets.extend([m.item() for m in mos_score.detach().cpu().numpy()])

                # 更新进度条显示当前损失
                pbar.set_postfix(loss=test_loss / (pbar.n + 1))

    # 计算 PLCC, SRCC, KLCC
    test_plcc, test_srcc, test_klcc = compute_metrics(test_predictions, test_targets)
    print(f"Test Loss: {test_loss / len(test_loader)}")
    print(f"Test PLCC: {test_plcc}, Test SRCC: {test_srcc}, Test KLCC: {test_klcc}")


if __name__ == "__main__":
    search_hyperparameters()
