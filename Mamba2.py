import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# 定义MambaBlock类
class MambaBlock(nn.Module):
    def __init__(self, d_model=3, d_inner=8, seq_len=10, d_result=1):
        super(MambaBlock, self).__init__()

        # 输入投影层：将输入映射到内部维度空间, 2 * d_inner 的原因是后续拆分成大小均为 d_inner的x和residual
        self.in_proj = nn.Linear(d_model, 2 * d_inner)

        # 增加层归一化层，帮助稳定训练
        self.layer_norm1 = nn.LayerNorm(d_inner)

        # 一维卷积层：处理序列数据，提取局部特征
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=3, padding=1)

        # 增加 Dropout 层，防止过拟合
        self.dropout = nn.Dropout(p=0.3)

        # 状态空间模型：建模时间序列关系
        self.ssm = StateSpaceModel(d_inner, seq_len)

        # 增加层归一化层，帮助稳定训练
        self.layer_norm2 = nn.LayerNorm(d_inner)

        # 输出投影层：映射回输入维度空间
        self.out_proj = nn.Linear(d_inner, d_inner)

        # 最后利用输出状态进行分类
        self.classifier = nn.Linear(d_inner * seq_len, d_result)

    def forward(self, x):
        b, l, d = x.shape  # 输入形状为 (batch_size, seq_len, feature_dim)

        # 输入投影，将输出分为两部分：x部分用于卷积层，res部分用于激活和相乘
        x, res = self.in_proj(x).chunk(2, dim=-1)  # 按照最后一个维度拆成两块，各是(b * l, d_inner)

        # 增加层归一化
        x = self.layer_norm1(x)

        # 将x的形状调整为适应卷积层的输入要求，并应用卷积和激活函数
        x = F.silu(self.conv1d(rearrange(x, 'b l d -> b d l')))

        # 卷积输出的形状调整回 (batch_size, seq_len, d_inner) 以匹配 SSM 模型输入
        x = rearrange(x, 'b d l -> b l d')

        # 增加 Dropout 防止过拟合
        x = self.dropout(x)

        # 状态空间模型处理后的结果与res的激活结果相乘
        y = self.ssm(x) * F.silu(res)

        # 增加层归一化
        y = self.layer_norm2(y)

        # 输出投影层，并加入跳跃连接
        y = self.out_proj(y) + x
        y = y.view(b, -1)

        # 输出投影层
        return y


# 状态空间模型类
class StateSpaceModel(nn.Module):
    def __init__(self, d_inner, seq_len):
        super(StateSpaceModel, self).__init__()

        # 状态转移、输入、输出矩阵参数
        self.deltaA = nn.Parameter(torch.randn(d_inner, d_inner))
        self.deltaB = nn.Parameter(torch.randn(d_inner, d_inner))
        self.C = nn.Parameter(torch.randn(d_inner, d_inner))

    def forward(self, x):
        b, l, d_in = x.shape

        # 初始化隐藏状态为全0，形状为 (batch_size, d_inner)
        hidden_state = torch.zeros((b, d_in), device=x.device)
        outputs = []

        for i in range(l):
            # 当前时间步输入
            x_t = x[:, i, :]  # 形状为 (batch_size, d_in)

            # 更新当前时间步的隐藏状态：A * h + B * x
            hidden_state = torch.matmul(hidden_state, self.deltaA) + torch.matmul(x_t, self.deltaB)

            # 计算当前时间步的输出：y = C * h
            output = torch.matmul(hidden_state, self.C)
            outputs.append(output)

        # 按照时间步，也就是seq_length这个维度，把输出堆叠起来，最终形状为 (batch_size, seq_len, d_in)
        return torch.stack(outputs, dim=1)
