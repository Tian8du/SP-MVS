import torch
import torch.fft
import torch.nn.functional as F

def cas_mvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    depth_loss = None

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')


        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * depth_loss
        else:
            total_loss += 1.0 * depth_loss

    return total_loss, depth_loss

# def cas_mvsnet_loss(inputs, depth_gt_ms, mask_ms, refined_depth=None, **kwargs):
#     depth_loss_weights = kwargs.get("dlossw", None)
#
#     total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
#     depth_loss = None
#     refined_loss = None  # 用来计算优化后的深度损失
#
#     for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
#         depth_est = stage_inputs["depth"]
#         depth_gt = depth_gt_ms[stage_key]
#         mask = mask_ms[stage_key]
#         mask = mask > 0.5
#
#         # 计算初步深度的损失
#         depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
#
#         # 计算深度优化后的损失（如果有提供优化后的深度）
#         if refined_depth is not None and stage_key == "stage3":  # 只对最后阶段进行优化
#             refined_depth_stage = refined_depth["stage3"]
#             refined_loss = F.smooth_l1_loss(refined_depth_stage[mask], depth_gt[mask], reduction='mean')
#
#         # 加入权重后累加损失
#         if depth_loss_weights is not None:
#             stage_idx = int(stage_key.replace("stage", "")) - 1
#             total_loss += depth_loss_weights[stage_idx] * depth_loss
#             if refined_loss is not None:
#                 total_loss += depth_loss_weights[stage_idx] * refined_loss
#         else:
#             total_loss += 1.0 * depth_loss
#             if refined_loss is not None:
#                 total_loss += 1.0 * refined_loss
#
#     return total_loss, depth_loss, refined_loss

def depth_distribution_similarity_loss(depth, depth_gt, mask, depth_min, depth_max):
    # depth_norm = depth * 128 / (depth_max - depth_min)[:,None,None]
    depth_norm = depth * 128 / (depth_max - depth_min)[:,None,None]
    # depth_norm = (depth - depth_min) * 128 / (depth_max - depth_min)[:,None,None]
    # depth_gt_norm = depth_gt * 128 / (depth_max - depth_min)[:,None,None]
    depth_gt_norm = depth_gt * 128 / (depth_max - depth_min)[:,None,None]
    # depth_gt_norm = (depth_gt - depth_min) * 128 / (depth_max - depth_min)[:,None,None]

    M_bins = 48
    kl_min = torch.min(torch.min(depth_gt), depth.mean()-3.*depth.std()).item()
    kl_max = torch.max(torch.max(depth_gt), depth.mean()+3.*depth.std()).item()
    bins = torch.linspace(kl_min, kl_max, steps=M_bins)

    kl_divs = []
    for i in range(len(bins) - 1):
        bin_mask = (depth_gt >= bins[i]) & (depth_gt < bins[i+1])
        merged_mask = mask & bin_mask

        if merged_mask.sum() > 0:
            p = depth_norm[merged_mask]
            # p_clamped = torch.clamp(p, min=1e-8)
            # 检查张量中是否有 NaN 值
            # contains_nan = torch.isnan(p).any().item()
            # if contains_nan:
            #     print("p中包含 NaN 值")
            q = depth_gt_norm[merged_mask]
            # 对 q 进行修正，确保不含零概率值
            # q_clamped = torch.clamp(q, min=1e-8)
            # contains_nan = torch.isnan(q).any().item()
            # if contains_nan:
            #     print("q中包含 NaN 值")
            # print('q',q)
            # kl_div = F.kl_div(torch.log(p_clamped)-torch.log(q_clamped), p, reduction='batchmean')
            # print('!!!!!!!!!1111',p.shape)
            # print('!!!!!!!!!2222',q.shape)
            kl_div = F.kl_div(F.log_softmax(p, dim=0)-F.log_softmax(q, dim=0), F.softmax(p, dim=0), reduction='batchmean')
            # print('1',kl_div)
            # contains_nan = torch.isnan(kl_div).any().item()
            # if contains_nan:
            #     print("kl_div中包含 NaN 值")
            # kl_div = torch.log(torch.clamp(kl_div, min=1e-8))
            kl_div = torch.log(torch.clamp(kl_div, min=1))
            # contains_nan = torch.isnan(kl_div).any().item()
            # if contains_nan:
            #     print("kl_div2中包含 NaN 值")
            # print('2',kl_div)
            kl_divs.append(kl_div)

    dds_loss = sum(kl_divs)
    return dds_loss

def STsatmvsloss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)
    # depth_loss_weights = kwargs.get("dlossw", [1, 1, 1])
    depth_values = kwargs.get("depth_values")
    # 获取 depth_values
    # depth_values = inputs[2]  # 假设 depth_values 是 inputs 的第三个元素
    # depth_values = inputs["depth_values"]
    # print('depth_values', depth_values)
    # print('inputs', inputs)
    # print('depth_values', type(depth_values))
    depth_min, depth_max = depth_values[:, 0], depth_values[:, -1]
    # print('min',depth_min.shape)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    dds_loss_stages = []
    depth_loss = None

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"]
        # depth_est = stage_inputs["depth_filtered"]
        depth_fre = stage_inputs["depth_filtered"]

        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
        # print(depth_fre)
        dds_loss = depth_distribution_similarity_loss(depth_fre, depth_gt, mask, depth_min, depth_max)
        # dds_loss = depth_distribution_similarity_loss(depth_est, depth_gt, mask, depth_min, depth_max)
        dds_loss_stages.append(dds_loss)

        # total loss
        lam1, lam2 = 0.8, 0.2
        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * (lam1 * depth_loss + lam2 * dds_loss)
        else:
            total_loss += 1.0 * (lam1 * depth_loss + lam2 * dds_loss)

    return total_loss, depth_loss


def cas_emvsnet_loss(ref_img, inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    depth_loss = 0.0
    edge_loss = 0.0  # 初始化边缘损失

    # 计算每个阶段的深度损失
    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        # 计算深度损失
        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * depth_loss
        else:
            total_loss += 1.0 * depth_loss

    # 计算边缘损失
    refine_est = inputs["refined_depth"]
    depth_gt = depth_gt_ms['stage3'].unsqueeze(1)
    mask = mask_ms['stage3'].unsqueeze(1)
    edge_loss, threshold = edge_aware_loss(ref_img, refine_est, depth_gt, mask)

    edge_loss = edge_loss * 0.2  # 边缘损失权重
    total_loss += edge_loss

    # 返回三个损失
    return total_loss, depth_loss, edge_loss


def compute_gradient(img):
    # Sobel kernels
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_x, sobel_y = sobel_x.to(img.device), sobel_y.to(img.device)

    grad_x = F.conv2d(img, sobel_x, padding=1)
    grad_y = F.conv2d(img, sobel_y, padding=1)

    gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    return gradient


def edge_aware_loss(img, pred_depth, gt_depth, mask, alpha=1.0):
    # 计算原始图像的梯度
    img_grad = compute_gradient(img)

    # Sobel X 和 Sobel Y 的卷积核大小为 3x3 或 5x5，但必须正确匹配
    sobel_x_5 = torch.tensor([[1, 0, -1, 0, 1], [2, 0, -2, 0, 2], [1, 0, -1, 0, 1], [2, 0, -2, 0, 2], [1, 0, -1, 0, 1]],
                             dtype=torch.float32).view(1, 1, 5, 5)
    sobel_y_5 = torch.tensor([[1, 2, 1, 0, 0], [2, 0, -2, 0, 0], [1, 0, -1, 0, 0], [2, 0, -2, 0, 0], [1, 0, -1, 0, 0]],
                             dtype=torch.float32).view(1, 1, 5, 5)

    sobel_x_5, sobel_y_5 = sobel_x_5.to(img.device), sobel_y_5.to(img.device)

    grad_x_5 = F.conv2d(img, sobel_x_5, padding=2)
    grad_y_5 = F.conv2d(img, sobel_y_5, padding=2)

    gradient_5 = torch.sqrt(grad_x_5 ** 2 + grad_y_5 ** 2)
    img_grad = (img_grad + gradient_5) / 2  # 多尺度梯度融合

    # 高斯平滑来去除噪声
    gaussian_kernel = torch.tensor(
        [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]],
        dtype=torch.float32).view(1, 1, 5, 5) / 256.0
    gaussian_kernel = gaussian_kernel.to(img.device)

    img_grad = F.conv2d(img_grad, gaussian_kernel, padding=2)

    # 计算阈值并生成边缘掩膜
    grad_mean = torch.mean(img_grad)
    grad_std = torch.std(img_grad)
    threshold = grad_mean + alpha * grad_std

    edge_mask = (img_grad > threshold).float()

    # 将mask与边缘掩膜结合
    combined_mask = (mask > 0.5).float() * edge_mask

    # 扩展维度使其与预测深度和真实深度兼容
    pred_depth = pred_depth.unsqueeze(1)  # [batch_size, 1, height, width]
    gt_depth = gt_depth.unsqueeze(1)  # [batch_size, 1, height, width]
    combined_mask = combined_mask.unsqueeze(1)  # [batch_size, 1, height, width]

    edge_loss = huber_loss(pred_depth[combined_mask > 0], gt_depth[combined_mask > 0], delta=1.0)
    edge_loss = edge_loss.mean()
    return edge_loss, threshold

def huber_loss(pred, target, delta=1.0):
    error = pred - target
    abs_error = torch.abs(error)
    condition = abs_error <= delta
    small_error_loss = 0.5 * error**2
    large_error_loss = delta * (abs_error - 0.5 * delta)
    return torch.where(condition, small_error_loss, large_error_loss)




def frequency_loss(depth_pred, depth_gt, mask, alpha=0.5):
    """
    基于频率域的综合损失函数
    :param depth_pred: 预测深度图 [B, 1, H, W]
    :param depth_gt: 真实深度图 [B, 1, H, W]
    :param mask: 有效像素掩膜 [B, 1, H, W]
    :param alpha: 高频损失的权重
    :return: 总损失, 高频损失, 低频损失
    """
    def frequency_split(image, threshold_ratio=0.1):
        # 傅里叶变换
        fft = torch.fft.fft2(image, dim=(-2, -1))
        fft_shift = torch.fft.fftshift(fft)

        # 分离高频和低频
        _, H, W = image.shape
        center_x, center_y = H // 2, W // 2
        radius = int(min(H, W) * threshold_ratio)
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        mask_low = ((x - center_x)**2 + (y - center_y)**2 <= radius**2).to(image.device)

        low_freq_fft = fft_shift * mask_low
        high_freq_fft = fft_shift * ~mask_low

        # 逆变换
        low_freq = torch.fft.ifft2(torch.fft.ifftshift(low_freq_fft), dim=(-2, -1)).real
        high_freq = torch.fft.ifft2(torch.fft.ifftshift(high_freq_fft), dim=(-2, -1)).real
        return low_freq, high_freq

    # 提取高频和低频
    low_freq_pred, high_freq_pred = frequency_split(depth_pred)
    low_freq_gt, high_freq_gt = frequency_split(depth_gt)

    # 有效像素掩膜
    mask = mask > 0.5

    # 计算高频和低频损失
    high_freq_loss = F.l1_loss(high_freq_pred[mask], high_freq_gt[mask], reduction='mean')
    low_freq_loss = F.l1_loss(low_freq_pred[mask], low_freq_gt[mask], reduction='mean')

    # 综合损失
    total_loss = alpha * high_freq_loss + (1 - alpha) * low_freq_loss
    return total_loss, high_freq_loss, low_freq_loss


def hyper_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)
    frequency_alpha = kwargs.get("freq_alpha", 0.5)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    depth_loss = None

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        # Smooth L1 Loss
        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

        # 频率损失
        freq_loss, _, _ = frequency_loss(depth_est, depth_gt, mask, alpha=frequency_alpha)

        # 权重组合
        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * (depth_loss + freq_loss)
        else:
            total_loss += 1.0 * (depth_loss + freq_loss)

    return total_loss, depth_loss

