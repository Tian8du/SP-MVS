import argparse
import datetime
from tensorboardX import SummaryWriter
import sys
from dataset import find_dataset_def
import torch.backends.cudnn as cudnn
from networks.casmvs import CascadeMVSNet
from networks.ucs import UCSNet
from networks.casred import CascadeREDNet
from networks.stsat import ST_SatMVS
from networks.emvs import CascadeEMVSNet
from networks.eucs import eUCSNet
from networks.epnet import EPNet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
from tools.utils import *
from networks.loss import cas_mvsnet_loss, STsatmvsloss, cas_emvsnet_loss, eucs_loss
from modules.Visual import  normalize_image, plot_all_images2
# torch.autograd.set_detect_anomaly(True)

"""
Author:Chen Liu
Department: Wuhan University
Date:2025-4-28
Version:0.1
email:sweet8degree@gmail.com
"""

# if the input size is fixed, the benchmark is true, else false
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='A PyTorch Implementation')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'pred'])
parser.add_argument('--model', default="casmvs", help='select model', choices=['samsat', 'red', "casmvs", "ucs", "emvs", "eucs","epnet"])
parser.add_argument('--geo_model', default="rpc", help='select dataset', choices=["rpc", "pinhole"])
parser.add_argument('--use_qc', default=False, help="whether to use Quaternary Cubic Form for RPC warping.")
parser.add_argument('--dataset_root', default=r'H:\MVS-Dataset\Test', help='dataset root')
parser.add_argument('--dataset_name', default=r'US3D', help='dataset name')
parser.add_argument('--place', default='JAX', choices=['JAX', 'OMA', 'JAX+OMA'], help='which place? OMA or JAX?')

# Resume and save parameters
parser.add_argument('--loadckpt', default=r"", help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints2', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', default=False, help='continue to train the model')

# input parameters
parser.add_argument('--view_num', type=int, default=3, help='Number of images.')
parser.add_argument('--ref_view', type=int, default=1)

parser.add_argument('--batch_size', type=int, default=1, help='train batch size')

# Cascade parameters
parser.add_argument('--ndepths', type=str, default="64,32,8", help='ndepths')
parser.add_argument('--min_interval', type=float, default=0.5, help='min_interval in the bottom stage')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--lamb', type=float, default=1.5, help="lamb in ucs-net")

parser.add_argument('--dlossw', type=str, default="0.5,1.0,2.0", help='depth loss weight for different stage')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')

# network architecture
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

# the param for training.
parser.add_argument('--lrepochs', type=str, default="8,10,15,20:2",
                    help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
parser.add_argument('--summary_freq', type=int, default=50, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed, default 42')
parser.add_argument('--gpu_id', type=str, default="0")

# if you want to train in bigger batch size,but don't have enough memory for GUP, try grad accumulation method!
parser.add_argument('--grad_acc', type=int, default=1, help="the step of grad accumulation")

# 1. parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)

# 2. show the device in use
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Current device: GPU")
else:
    device = torch.device("cpu")
    print("Current device: CPU")

# 3. set the train and test path
trainpath = args.dataset_root
testpath = args.dataset_root

# 4. judge whether resume
if args.resume:
    assert args.mode == "train"
    assert args.loadckpt is None
if testpath is None:
    testpath = trainpath

# 5. set the seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# 6. set the save dir, create logger for mode "train" and "testall"
cur_log_dir = os.path.join(args.logdir, "{}/{}".format(args.model, args.geo_model)).replace("\\", "/")
ck_dir = os.path.join(cur_log_dir, "train").replace("\\", "/")
if not os.path.exists(ck_dir):
    os.makedirs(ck_dir)
if args.mode == "train":
    if not os.path.isdir(cur_log_dir):
        os.makedirs(cur_log_dir)
    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)
    print("creating new summary file")
    logger = SummaryWriter(cur_log_dir)

# 7. accumulation grad method
accumulation_steps = args.grad_acc
if accumulation_steps != 1:
    print("Use Grad_accumulation Method")
    args.lr = args.lr * accumulation_steps

# 8. dataset, dataloader
MVSDataset = find_dataset_def(args.geo_model, args.dataset_name)
train_dataset = MVSDataset(trainpath, "train", args.view_num, ref_view=args.ref_view, use_qc=args.use_qc)
test_dataset = MVSDataset(testpath, "test", args.view_num, ref_view=args.ref_view, use_qc=args.use_qc)

height_range = None
if args.place == "JAX":
    height_range = [-32, 224]
elif args.place == "OMA":
    height_range = [128, 384]
elif args.place == "JAX+OMA":
    height_range = [-32, 384]
# === 新增统一固定height range设置 ===
train_dataset.use_fixed_height_range = True
train_dataset.fixed_height_range = height_range
test_dataset.use_fixed_height_range = True
test_dataset.fixed_height_range = height_range
# =====================================

TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

# 9. model
model = None
if args.model == "samsat":
    model = ST_SatMVS(min_interval=args.min_interval,
                          ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                          geo_model=args.geo_model, use_qc=args.use_qc)
elif args.model == "casmvs":
    model = CascadeMVSNet(min_interval=args.min_interval,
                          ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                          geo_model=args.geo_model, use_qc=args.use_qc)
    print("===============> Model: Cascade MVS Net ===========>")
elif args.model == "ucs":
    model = UCSNet(lamb=args.lamb, stage_configs=[int(nd) for nd in args.ndepths.split(",") if nd],
                   base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                   geo_model=args.geo_model, use_qc=args.use_qc)
    print("===============> Model: UCS-Net ===========>")
elif args.model == "red":
    model = CascadeREDNet(min_interval=args.min_interval,
                          ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                          geo_model=args.geo_model, use_qc=args.use_qc)
    print("===============> Model: Cascade RED Net ===========>")
elif args.model == "emvs":
    model = CascadeEMVSNet(min_interval=args.min_interval,
                          ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                          geo_model=args.geo_model, use_qc=args.use_qc)
    print("===============> Model: Cascade EMVS Net ===========>")
elif args.model == "eucs":
    model = eUCSNet(lamb=args.lamb, stage_configs=[int(nd) for nd in args.ndepths.split(",") if nd],
                   base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                   geo_model=args.geo_model, use_qc=args.use_qc)
    print("===============> Model: eUCS-Net ===========>")
elif args.model == "epnet":
    model = EPNet(min_interval=args.min_interval,
                  ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                   depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i]
                  )
    print("===============> Model: Ep-Net ===========>")
else:
    raise Exception("{}? Not implemented yet!".format(args.model))

# 10. CUDA number
if torch.cuda.device_count() > 1:
    print("Using multiple GPUs for training")
    model = nn.DataParallel(model)  # Enable multi-GPU parallelism
else:
    print("Using a single GPU or CPU for training")

model = model.cuda() if torch.cuda.is_available() else model

# 11. choose the loss
if args.model == "casmvs":
    model_loss = cas_mvsnet_loss
elif args.model == "samsat":
    model_loss = STsatmvsloss
elif args.model == "ucs":
    model_loss = cas_mvsnet_loss
elif args.model == "red":
    model_loss = cas_mvsnet_loss
elif args.model == "emvs":
    model_loss = cas_emvsnet_loss
elif args.model == "eucs":
    model_loss = eucs_loss
elif args.model == "epnet":
    model_loss = cas_mvsnet_loss
else:
    model_loss = cas_mvsnet_loss

# 12. initial optimizer
optimizer = optim.RMSprop([{'params': model.parameters(), 'initial_lr': args.lr}],
                          lr=args.lr, alpha=0.9, weight_decay=args.wd)

# 13. load parameters
start_epoch = 1
if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
    saved_models = [fn for fn in os.listdir(cur_log_dir) if fn.endswith(".ckpt") and len(fn.split("_")) == 2]
    # print(saved_models)
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[1].split(".")[0]))
    # use the latest checkpoint file
    # print(saved_models)
    load_ckpt = os.path.join(cur_log_dir, saved_models[-1])
    print("resuming", load_ckpt)
    state_dict = torch.load(load_ckpt, weights_only=False)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = int(saved_models[-1].split("_")[1].split(".")[0]) + 1
    # print(saved_models)
elif args.loadckpt:
    # load checkpoint file specified by args.load_ckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])

# # 加载原始部分的模型
# pretrained_dict = torch.load(r'F:\Codes2\DC-SatMVS-main\checkpoints\eucs\rpc\model_000013.ckpt')
# # 获取新模型的当前权重字典
# model_dict = model.state_dict()
# # 只保留匹配的层
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
# # 更新现有模型的状态字典
# model_dict.update(pretrained_dict)
# # 加载更新后的权重
# model.load_state_dict(model_dict)

print("start at epoch {}".format(start_epoch))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


# 14.main function
def train():
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    # 这个步骤很多人都写错了, lr_scheduler不应该在 epoch_idx内部调整
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=-1)

    for epoch_idx in range(start_epoch, args.epochs+1):
        print('Epoch {}:'.format(epoch_idx))
        # 输出当前学习率
        current_lr = lr_scheduler.get_last_lr()[0]  # 获取当前学习率
        print(f'Current learning rate: {current_lr}')

        global_step = len(TrainImgLoader) * epoch_idx

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):

            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0

            model.train()
            optimizer.zero_grad()
            loss, scalar_outputs, image_outputs = train_sample(sample, detailed_summary=do_summary)
            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()  # 更新参数
                optimizer.zero_grad()
                # lr_scheduler.step()
                # current_lr = lr_scheduler.get_last_lr()[0]  # 获取当前学习率
                # print(f'Current learning rate: {current_lr}')

            loss = round(tensor2float(loss), 4)
            scalar_outputs = tensor2float(scalar_outputs)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            print(
                'Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}, train_result = {}'.format(
                    epoch_idx, args.epochs, batch_idx, len(TrainImgLoader), loss,
                    time.time() - start_time, scalar_outputs))
            del scalar_outputs, image_outputs

        # 应该训练完毕一个epoch再改变lr_scheduler.
        lr_scheduler.step()
        # testing
        avg_test_scalars = DictAverageMeter()
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)

            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}, {}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                     len(TestImgLoader), loss,
                                                                                     time.time() - start_time, scalar_outputs))

            del scalar_outputs, image_outputs
        save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
        print("avg_test_scalars:", avg_test_scalars.mean())
        abs_depth_error = avg_test_scalars.mean()["abs_depth_acc"]

        train_record = open(cur_log_dir + '/train_record.txt', "a+")
        train_record.write(str(epoch_idx) + ' ' + str(avg_test_scalars.mean()) + '\n')
        train_record.close()

        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(cur_log_dir, epoch_idx, abs_depth_error))
        # gc.collect()alars:", avg_test_scalars.mean())
        # gc.collect()


def test():
    # create output folder
    output_folder = os.path.join(testpath, 'height_result')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    avg_test_scalars = DictAverageMeter()

    total_time = 0
    for batch_idx, sample in enumerate(TestImgLoader):

        bview = sample['out_view'][0]
        bname = sample['out_name'][0]

        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=True)
        avg_test_scalars.update(scalar_outputs)
        scalar_outputs = {k: float("{0:.6f}".format(v)) for k, v in scalar_outputs.items()}
        total_time += time.time() - start_time
        print("Iter {}/{}, {}, time = {:3f}, test results = {}".format(batch_idx, len(TestImgLoader),
                                                                       bname, time.time() - start_time, scalar_outputs))

        # save results
        depth_est = np.float32(np.squeeze(tensor2numpy(image_outputs["depth_est"])))
        prob = np.float32(np.squeeze(tensor2numpy(image_outputs["photometric_confidence"])))


        depth_gt = sample['depth']['stage3']
        mask = sample['mask']['stage3']

        depth_gt = np.float32(np.squeeze(tensor2numpy(depth_gt)))
        mask = (np.squeeze(tensor2numpy(mask))).astype(int)

        depth_gt[mask < 0.5] = -999.0

        first_image = np.float32(np.squeeze(tensor2numpy(image_outputs["ref_img"])))

        # plot_depth_comparison(depth_gt, depth_est)
        plot_all_images2(depth_gt, depth_est, normalize_image(first_image), prob)

        # plt.imshow(depth_est)
        # plt.show()

        del scalar_outputs, image_outputs

    print("final, time = {:3f}, test results = {}".format(total_time, avg_test_scalars.mean()))


def train_sample(sample, detailed_summary=False):
    # print("img", sample["out_name"])
    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]
    # ratio = (mask == 1).float().mean().item()
    # print(f"MASK Valid Ratio: {ratio:.4f}")


    outputs = model(sample_cuda["imgs"], sample_cuda["cam_para"], sample_cuda["depth_values"])
    if args.model == "samsat":
        depth_est = outputs["stage3"]["depth_filtered"]
    elif args.model == "emvs":
        # depth_est = outputs["refined_depth"].squeeze(0)
        depth_est = outputs["stage3"]["depth"]

    else:
        depth_est = outputs["stage3"]["depth"]


    ref_img = sample_cuda["imgs"][:, 0]
    if args.model == "emvs":
        loss, depth_loss, edge_loss = model_loss(ref_img, outputs, depth_gt_ms, mask_ms, dlossw=[float(e) for e in args.dlossw.split(",") if e], depth_values=sample_cuda["depth_values"])
    else:
        loss, depth_loss = model_loss(outputs, depth_gt_ms, mask_ms, dlossw=[float(e) for e in args.dlossw.split(",") if e], depth_values=sample_cuda["depth_values"])


    if args.model == "emvs":
        scalar_outputs = {"loss": loss, "depth_loss": depth_loss, "edge_loss": edge_loss}
    else:
        scalar_outputs = {"loss": loss, "depth_loss": depth_loss}
    image_outputs = {"depth_est": depth_est, "depth_gt": depth_gt,
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]["stage1"]}

    if detailed_summary:
        image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask
        scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, 250.0)
        scalar_outputs["RMSE"] = RMSE_metrics(depth_est, depth_gt, mask > 0.5, 250.0)
        scalar_outputs["thres1.0m_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 1.0)
        scalar_outputs["thres2.5m_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2.5)
        scalar_outputs["thres7.5m_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 7.5)


    return loss, scalar_outputs, image_outputs


@make_nograd_func
def test_sample(sample, detailed_summary=True):
    model.eval()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    outputs = model(sample_cuda["imgs"], sample_cuda["cam_para"], sample_cuda["depth_values"])

    if args.model == "samsat":
        depth_est = outputs["stage3"]["depth_filtered"]
    elif args.model == "emvs":
        depth_est = outputs["refined_depth"].squeeze(0)
    elif args.model == "ucs":
        depth_est = outputs["stage3"]["depth"]
    else:
        # depth_est = outputs["refined_depth"].squeeze(0)
        # depth_est = outputs["stage3"]["depth"]
        depth_est = outputs["depth"]

    photometric_confidence = outputs["stage3"]["photometric_confidence"]

    if args.model == "emvs":
        loss, depth_loss, edge_loss = model_loss(sample_cuda["imgs"][:, 0], outputs, depth_gt_ms, mask_ms,
                                                 dlossw=[float(e) for e in args.dlossw.split(",") if e],
                                                 depth_values=sample_cuda["depth_values"])
    else:
        loss, depth_loss = model_loss(outputs, depth_gt_ms, mask_ms,
                                      dlossw=[float(e) for e in args.dlossw.split(",") if e],
                                      depth_values=sample_cuda["depth_values"])

    if args.model == "emvs":
        scalar_outputs = {"loss": loss, "depth_loss": depth_loss, "edge_loss": edge_loss}
    else:
        scalar_outputs = {"loss": loss, "depth_loss": depth_loss}
    image_outputs = {"depth_est": depth_est,
                     "photometric_confidence": photometric_confidence,
                     "depth_gt": sample["depth"]["stage1"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]["stage1"]}

    if detailed_summary:
        image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask

    scalar_outputs["abs_depth_acc"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, 250.0)
    scalar_outputs["RMSE"] = RMSE_metrics(depth_est, depth_gt, mask > 0.5, 250.0)
    scalar_outputs["1.0m_acc"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 1.0)  #0.6
    scalar_outputs["2.5m_acc"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2.5)
    scalar_outputs["7.5m_acc"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 7.5)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


if __name__ == '__main__':
    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()

