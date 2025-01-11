import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)
def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return
def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAMBlock, self).__init__()
        self.channel_attention = SEBlock(in_channels, reduction_ratio)
        self.spatial_attention = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        # Channel Attention
        x = self.channel_attention(x)

        # Spatial Attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        spatial_attention_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_attention_map = self.spatial_attention(spatial_attention_input)
        return x * spatial_attention_map


class DynamicSobelKernel(nn.Module):
    def __init__(self, in_channels):
        super(DynamicSobelKernel, self).__init__()
        # 初始化四个方向的动态卷积核
        self.kernel_x = nn.Parameter(
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).expand(in_channels,
                                                                                                            1, 3, 3))
        self.kernel_y = nn.Parameter(
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).expand(in_channels,
                                                                                                            1, 3, 3))
        self.kernel_45 = nn.Parameter(
            torch.tensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=torch.float32).view(1, 1, 3, 3).expand(in_channels,
                                                                                                            1, 3, 3))
        self.kernel_135 = nn.Parameter(
            torch.tensor([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3).expand(in_channels,
                                                                                                            1, 3, 3))

        # 动态权重
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.delta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # 计算四个方向的梯度
        grad_x = F.conv2d(x, self.kernel_x, padding=1, groups=x.size(1))
        grad_y = F.conv2d(x, self.kernel_y, padding=1, groups=x.size(1))
        grad_45 = F.conv2d(x, self.kernel_45, padding=1, groups=x.size(1))
        grad_135 = F.conv2d(x, self.kernel_135, padding=1, groups=x.size(1))

        # 动态加权融合
        edge = torch.sqrt(self.alpha * grad_x ** 2 + self.beta * grad_y ** 2 +
                          self.gamma * grad_45 ** 2 + self.delta * grad_135 ** 2)
        return edge

class EdgeDetectionModule(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(EdgeDetectionModule, self).__init__()

        # Sobel 卷积核，用于计算 X 和 Y 方向的梯度
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        # Sobel卷积操作
        self.sobel_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)

        # 使用固定的 Sobel 卷积核
        self.sobel_x.weight.data = sobel_kernel_x.expand(in_channels, 1, 3, 3)
        self.sobel_y.weight.data = sobel_kernel_y.expand(in_channels, 1, 3, 3)

        # 禁用 Sobel 卷积核的梯度计算
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False

        # 激活函数
        self.relu = nn.ReLU(inplace=True)

        # 1x1 卷积调整通道数
        self.adjust_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 是否下采样
        self.downsample = downsample
        if self.downsample:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # 计算 X 和 Y 方向的梯度
        x = x[:, 0:1, :, :]
        grad_x = self.sobel_x(x)
        grad_y = self.sobel_y(x)

        # 计算梯度幅值，确保非原地操作
        edge = torch.sqrt(torch.pow(grad_x.detach(), 2) + torch.pow(grad_y.detach(), 2))
        edge = self.relu(edge)

        # 调整通道数
        out = self.adjust_conv(edge)

        # 下采样
        if self.downsample:
            out = self.pool(out)

        return out

def gaussian_blur(input_tensor, kernel_size=5, sigma=1.0):
    # 创建高斯核
    kernel = torch.tensor([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ], dtype=torch.float32) / 256.0  # 对高斯核归一化

    # 在 PyTorch 中需要创建一个卷积核来实现模糊
    kernel = kernel.view(1, 1, 5, 5).expand(input_tensor.size(1), 1, 5, 5).to(input_tensor.device)

    # 使用卷积进行高斯模糊
    blurred = F.conv2d(input_tensor, kernel, padding=2, groups=input_tensor.size(1))

    return blurred

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze操作：按通道维度求平均池化
        b, c, _, _ = x.size()
        squeeze = F.adaptive_avg_pool2d(x, 1).view(b, c)
        # Excitation操作：通过全连接层生成注意力系数
        excitation = self.fc1(squeeze)
        excitation = F.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation).view(b, c, 1, 1)
        # 进行通道上的加权
        return x * excitation

# 边缘注意力分支
class EdgeAttentionBranch(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(EdgeAttentionBranch, self).__init__()
        self.edge_cnn = EdgeDetectionModule(in_channels, out_channels, downsample)  # 边缘特征提取网络
        self.sigmoid = nn.Sigmoid()  # 生成注意力图

    def forward(self, x):
        edge_features = self.edge_cnn(x)  # 提取边缘特征
        attention_map = self.sigmoid(edge_features)  # 将边缘特征映射到注意力图
        return attention_map

class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=in_channels, bias=False)  # Depthwise 卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # Pointwise 卷积

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class FusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(FusionModule, self).__init__()
        self.main_branch = HybridConvModule(in_channels, out_channels, downsample)
        self.edge_attention_branch = EdgeAttentionBranch(in_channels, out_channels, downsample)
        self.cbam_block = CBAMBlock(out_channels)  # 使用 CBAM 注意力机制

    def forward(self, x):
        # 提取主分支特征和边缘注意力图
        main_features = self.main_branch(x)
        edge_attention = self.edge_attention_branch(x)

        # 将注意力图应用于主分支特征
        enhanced_features = main_features * edge_attention

        # 使用 CBAM 模块增强特征
        enhanced_features = self.cbam_block(enhanced_features)

        return enhanced_features



class HybridConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(HybridConvModule, self).__init__()
        self.downsample = downsample

        if self.downsample:
            self.conv1 = DepthwiseConv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        else:
            self.conv1 = DepthwiseConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 多尺度空洞卷积
        self.dilated_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.dilated_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=4, dilation=4)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 深度卷积
        self.depthwise_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # 1x1卷积
        self.adjust_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2_1 = self.dilated_conv1(x1)
        x2_2 = self.dilated_conv2(x1)
        x2 = self.relu(self.bn2(x2_1 + x2_2))
        x3 = self.relu(self.bn3(self.depthwise_conv(x2)))

        # 拼接融合
        fused = torch.cat([x1, x2, x3], dim=1)
        out = self.adjust_conv(fused)

        return out




class DeConv2dFuse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(DeConv2dFuse, self).__init__()

        self.deconv = Deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                               bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2d(2*out_channels, out_channels, kernel_size, stride=1, padding=1,
                           bn=bn, relu=relu, bn_momentum=bn_momentum)

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x

class FeatExt3_ref(nn.Module):
    # There should be reference image feature extra network
    def __init__(self):
        super(FeatExt3_ref, self).__init__()
        base_channels = 8
        # In Image and the extra channel
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU())

        self.conv0_0 = Conv2d(base_channels * 1, base_channels * 1, 3, stride=1, padding=1)
        self.conv0_1 = Conv2d(base_channels * 1, base_channels * 1, 3, stride=1, relu=False, padding=1)
        self.conv0_2 = Conv2d(base_channels * 1, base_channels * 1, 1, stride=1, relu=False)

        self.conv1_0 = Conv2d(base_channels * 1, base_channels * 1, 3, stride=1, padding=1)
        self.conv1_1 = Conv2d(base_channels * 1, base_channels * 1, 3, stride=1, relu=False, padding=1)

        self.conv2_0 = Conv2d(base_channels * 1, base_channels * 2, 3, stride=2, padding=1)
        self.conv2_1 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)
        self.conv2_2 = Conv2d(base_channels * 1, base_channels * 2, 1, stride=2, relu=False)

        self.conv3_0 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, padding=1)
        self.conv3_1 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)

        self.conv4_0 = Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.conv4_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)
        self.conv4_2 = Conv2d(base_channels * 2, base_channels * 4, 1, stride=2, relu=False)

        self.conv5_0 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, padding=1)
        self.conv5_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)

        self.conv6_0 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, 2, 1, 1, bias=False)
        self.conv6_1 = nn.Conv2d(base_channels * 4, base_channels * 2, 3, 1, 1, bias=False)
        self.conv6_2 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, padding=1)
        self.conv6_3 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)

        self.final_conv_1 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.final_conv_2 = nn.Conv2d(16, 16, 3, 1, 1, bias=False)

    def forward(self, x):
        x = self.init_conv(x)

        residual = x
        x = self.conv0_1(self.conv0_0(x))
        x += self.conv0_2(residual)
        x = nn.ReLU(inplace=True)(x)

        residual = x
        x = self.conv1_1(self.conv1_0(x))
        x += residual
        out3 = nn.ReLU(inplace=True)(x)


        residual = out3
        x = self.conv2_1(self.conv2_0(out3))
        x += self.conv2_2(residual)
        x = nn.ReLU(inplace=True)(x)
        residual = x
        x = self.conv3_1(self.conv3_0(x))
        x += residual
        out2 = nn.ReLU(inplace=True)(x)

        residual = out2
        x = self.conv4_1(self.conv4_0(out2))
        x += self.conv4_2(residual)
        x = nn.ReLU(inplace=True)(x)
        residual = x
        x = self.conv5_1(self.conv5_0(x))
        x += residual
        out1 = nn.ReLU(inplace=True)(x)

        x = self.conv6_0(out1)
        x = torch.cat([x, out2], 1)
        x = self.conv6_1(x)
        residual = x
        x = self.conv6_3(self.conv6_2(x))
        x += residual
        out2 = nn.ReLU(inplace=True)(x)

        outputs = {}
        outputs["stage1"] = self.final_conv_1(out1)
        outputs["stage2"] = self.final_conv_2(out2)
        outputs["stage3"] = out3

        return outputs


class Refinement(nn.Module):
    def __init__(self, feat_channels):
        super(Refinement, self).__init__()
        base_channels = 8

        self.conv1_0 = nn.Sequential(
            Conv2d(1, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1))

        self.conv1_2 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, 1, 1)

        self.conv2_0 = Conv2d(feat_channels + base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.conv2_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)
        self.conv2_2 = Conv2d(feat_channels + base_channels * 2, base_channels * 4, 1, stride=2, relu=False)

        self.conv3_0 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, padding=1)
        self.conv3_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)

        self.conv4_0 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, 2, 1, 1, bias=False)
        self.conv4_1 = nn.Conv2d(feat_channels + base_channels * 4, base_channels * 2, 3, 1, 1, bias=False)
        self.conv4_2 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, padding=1)
        self.conv4_3 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)

        self.final_conv = nn.Conv2d(base_channels * 2, 1, 3, padding=1, bias=False)

    def forward(self, depth, img_feat):
        depth_mean = torch.mean(depth.reshape(depth.shape[0], -1), -1, keepdim=True)
        depth_std = torch.std(depth.reshape(depth.shape[0], -1), -1, keepdim=True)
        depth = (depth.unsqueeze(1) - depth_mean.unsqueeze(-1).unsqueeze(-1)) / depth_std.unsqueeze(-1).unsqueeze(-1)
        depth_min, _ = torch.min(depth.reshape(depth.shape[0], -1), -1, keepdim=True)
        depth_max, _ = torch.max(depth.reshape(depth.shape[0], -1), -1, keepdim=True)

        # 提取深度图特征，变换通道数目
        depth_feat = self.conv1_2(self.conv1_0(depth))
        # 深度图特征与原图特征Cat
        cat = torch.cat((img_feat, depth_feat), dim=1)
        # print(depth_feat.shape)
        # print(cat.shape)


        residual = cat
        x = self.conv2_1(self.conv2_0(cat))
        x += self.conv2_2(residual)
        x = nn.ReLU(inplace=True)(x)
        residual = x
        x = self.conv3_1(self.conv3_0(x))
        x += residual
        out1 = nn.ReLU(inplace=True)(x)
        # out1 应该是残差提取模块后的特征

        x = self.conv4_0(out1)
        x = torch.cat([x, cat], 1)
        x = self.conv4_1(x)
        residual = x
        x = self.conv4_3(self.conv4_2(x))
        x += residual
        out2 = nn.ReLU(inplace=True)(x)

        res = self.final_conv(out2)

        res_ = torch.zeros_like(res)
        for i in range(res.shape[0]):
            res_[i] = torch.clamp(res[i], min=depth_min[i].cpu().item(), max=depth_max[i].cpu().item())
        depth = (res_ +depth * depth_std.unsqueeze(-1).unsqueeze(-1) + depth_mean.unsqueeze(-1).unsqueeze(-1))

        return depth.squeeze(1)


class FeatureNet2(nn.Module):
    def __init__(self, base_channels, num_stage=3, stride=4, arch_mode="unet+sobel"):
        super(FeatureNet2, self).__init__()
        assert arch_mode in ["unet+sobel"], print("mode must be in 'unet' or 'fpn', but get:{}".format(arch_mode))
        print("*************feature extraction arch mode:{}****************".format(arch_mode))
        self.arch_mode = arch_mode
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            FusionModule(1, base_channels, downsample=False),
            # HybridConvModule(3, base_channels, downsample=False),
        )

        self.conv1 = nn.Sequential(
            # FusionModule(base_channels, base_channels * 2, downsample=True),
            # Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            # HybridConvModule(base_channels, base_channels * 2, downsample=True),
            # HybridConvModule(base_channels * 2, base_channels * 2),
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            # FusionModule(base_channels * 2, base_channels * 4, downsample=True),
            # Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            # HybridConvModule(base_channels * 2, base_channels * 4, downsample=True),
            # HybridConvModule(base_channels * 4, base_channels * 4),
            Conv2d(base_channels* 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]


        if num_stage == 3:
            self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)
            self.deconv2 = DeConv2dFuse(base_channels * 2, base_channels, 3)

            self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
            self.out3 = nn.Conv2d(base_channels, base_channels, 1, bias=False)
            self.out_channels.append(2 * base_channels)
            self.out_channels.append(base_channels)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        # print('conv2',conv2.shape)

        intra_feat = conv2
        outputs = {}
        out = self.out1(intra_feat)
        # print('out', out.shape)
        outputs["stage1"] = out

        if self.num_stage == 3:
            intra_feat = self.deconv1(conv1, intra_feat)
            out = self.out2(intra_feat)
            outputs["stage2"] = out

            intra_feat = self.deconv2(conv0, intra_feat)
            out = self.out3(intra_feat)
            outputs["stage3"] = out

        return outputs


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class RefineNet2(nn.Module):
    def __init__(self):
        super(RefineNet2, self).__init__()
        self.conv1 = ConvBnReLU(2, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)

        # 多尺度卷积分支
        self.conv3_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)

        # 新增的卷积层，用于生成多尺度加权系数
        self.conv_weight1 = nn.Conv2d(32, 32, kernel_size=1, padding=0)
        self.conv_weight2 = nn.Conv2d(32, 32, kernel_size=1, padding=0)

        # 深度残差预测
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        concat = torch.cat((img, depth_init), dim=1)  # 拼接图像和初始深度图

        x1 = self.conv1(concat)  # 特征提取
        x2 = self.conv2(x1)

        # 使用多尺度卷积
        x3_1 = self.conv3_1(x2)
        x3_2 = self.conv3_2(x2)

        # 加权融合不同尺度特征
        weight_1 = torch.sigmoid(self.conv_weight1(x2))
        weight_2 = torch.sigmoid(self.conv_weight2(x2))
        x3 = weight_1 * x3_1 + weight_2 * x3_2  # 基于权重加权融合

        depth_residual = self.res(x3)  # 预测深度残差
        depth_refined = depth_init + depth_residual  # 深度优化
        return depth_refined



# Use mamba vision to extra features
class FeatureNet3(nn.Module):
    def __init__(self, base_channels, num_stage=3, stride=4, arch_mode="unet+sobel"):
        super(FeatureNet3, self).__init__()
        assert arch_mode in ["unet+sobel"], print("mode must be in 'unet' or 'fpn', but get:{}".format(arch_mode))
        print("*************feature extraction arch mode:{}****************".format(arch_mode))
        self.arch_mode = arch_mode
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            FusionModule(1, base_channels, downsample=False),
            # HybridConvModule(3, base_channels, downsample=False),
        )

        self.conv1 = nn.Sequential(
            # FusionModule(base_channels, base_channels * 2, downsample=True),
            HybridConvModule(base_channels, base_channels * 2, downsample=True),
            # HybridConvModule(base_channels * 2, base_channels * 2),
        )

        self.conv2 = nn.Sequential(
            # FusionModule(base_channels * 2, base_channels * 4, downsample=True),
            HybridConvModule(base_channels * 2, base_channels * 4, downsample=True),
            # HybridConvModule(base_channels * 4, base_channels * 4),
        )

        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]


        if num_stage == 3:
            self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)
            self.deconv2 = DeConv2dFuse(base_channels * 2, base_channels, 3)

            self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
            self.out3 = nn.Conv2d(base_channels, base_channels, 1, bias=False)
            self.out_channels.append(2 * base_channels)
            self.out_channels.append(base_channels)

        elif num_stage == 2:
            self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)

            self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
            self.out_channels.append(2 * base_channels)


    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        # print('conv2',conv2.shape)

        intra_feat = conv2
        outputs = {}
        out = self.out1(intra_feat)
        # print('out', out.shape)
        outputs["stage1"] = out

        if self.num_stage == 3:
            intra_feat = self.deconv1(conv1, intra_feat)
            out = self.out2(intra_feat)
            outputs["stage2"] = out

            intra_feat = self.deconv2(conv0, intra_feat)
            out = self.out3(intra_feat)
            outputs["stage3"] = out

        elif self.num_stage == 2:
            intra_feat = self.deconv1(conv1, intra_feat)
            out = self.out2(intra_feat)
            outputs["stage2"] = out


        return outputs
