import torch
import torch.nn as nn
import torch.nn.functional as F
# from mamba_ssm import Mamba


class TriDirectionalScanner(nn.Module):
    """三向扫描序列化模块（基于网页1/5/9）"""

    def __init__(self, dim):
        super().__init__()
        # 三向特征投影
        self.proj_x = nn.Conv3d(dim, dim // 3, 1)
        self.proj_y = nn.Conv3d(dim, dim // 3, 1)
        self.proj_z = nn.Conv3d(dim, dim // 3, 1)
        # 动态权重生成（网页5）
        self.weights_gen = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dim, 3, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        B, C, D, H, W = x.shape

        # X/Y/Z轴交替扫描（网页1的Z-order改进）
        x_seq_x = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, D, C)  # X优先
        x_seq_y = x.permute(0, 2, 4, 1, 3).reshape(B * D * W, H, C)  # Y优先
        x_seq_z = x.permute(0, 2, 3, 1, 4).reshape(B * D * H, W, C)  # Z优先

        # 动态权重融合（网页5注意力机制）
        weights = self.weights_gen(x).squeeze()
        return (weights[:, 0:1] * self.proj_x(x_seq_x) +
                weights[:, 1:2] * self.proj_y(x_seq_y) +
                weights[:, 2:3] * self.proj_z(x_seq_z))


class DeformablePositionEmbed(nn.Module):
    """可变形位置编码（网页9/10）"""

    def __init__(self, dim):
        super().__init__()
        self.offset_net = nn.Sequential(
            nn.Conv3d(dim, dim // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(dim // 4, 3, 3, padding=1)
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        # 生成偏移场
        offsets = self.offset_net(x).permute(0, 2, 3, 4, 1)
        # 构建可变形网格
        grid = self._create_grid(D, H, W).to(x.device).expand(B, -1, -1, -1, -1)
        return F.grid_sample(grid, offsets, align_corners=False)

    def _create_grid(self, D, H, W):
        return torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, D),
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W)
        ), dim=-1)


class MambaBottleneck(nn.Module):
    """改进的Bottleneck模块，仅需替换原conv5+conv6"""
    def __init__(self, in_dim):
        super().__init__()
        # 三维序列化
        self.scanner = TriDirectionalScanner(in_dim)
        # 状态空间模型
        self.mamba = Mamba(d_model=in_dim, d_state=64, d_conv=4, expand=2)
        # 动态位置编码
        self.deform_pe = DeformablePositionEmbed(in_dim)
        # 通道增强
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_dim*2, in_dim//4, 1),
            nn.GELU(),
            nn.Conv3d(in_dim//4, in_dim*2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        # 序列化处理（保持三维拓扑）
        x_seq = self.scanner(x)  # [B*D*H*W, C]
        # 动态位置编码
        pos_emb = self.deform_pe(x).view(B*D*H*W, C)
        # Mamba全局建模
        global_feat = self.mamba(x_seq + pos_emb)
        # 通道注意力增强
        feat_3d = global_feat.view(B, D, H, W, -1).permute(0,4,1,2,3)
        channel_weights = self.channel_att(feat_3d)
        return feat_3d * channel_weights

class MambaCostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        # 保持原始编码器
        self.conv0 = nn.Conv3d(in_channels, base_channels, 3, padding=1)
        self.conv1 = nn.Conv3d(base_channels, base_channels*2, 3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(base_channels*2, base_channels*2, 3, padding=1)
        self.conv3 = nn.Conv3d(base_channels*2, base_channels*4, 3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(base_channels*4, base_channels*4, 3, padding=1)

        # Bottleneck替换为Mamba模块
        self.bottleneck = MambaBottleneck(base_channels*4)  # 替换原conv5+conv6

        # 保持原始解码器
        self.conv7 = nn.ConvTranspose3d(base_channels*8, base_channels*4, 3,
                                      stride=2, padding=1, output_padding=1)
        self.conv9 = nn.ConvTranspose3d(base_channels*4, base_channels*2, 3,
                                      stride=2, padding=1, output_padding=1)
        self.conv11 = nn.ConvTranspose3d(base_channels*2, base_channels, 3,
                                       stride=2, padding=1, output_padding=1)
        self.prob = nn.Conv3d(base_channels, 1, 3, padding=1)