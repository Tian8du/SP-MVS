import json
from torch.utils.data import Dataset
from dataset.data_io import *
from dataset.preprocess import *
from dataset.gen_list import *
import copy
import rasterio

# This is the dataset for US3D-MVS, DeepLearning.
class US3DDataset(Dataset):
    """
    """
    def __init__(self, data_folder, mode, view_num, ref_view=1, use_qc=False):
        """
        the Initial
        Args:
            data_folder: the folder path for this dataset
            mode:  the mode for training
            view_num:  the number of multiple satellite images
            ref_view:  the reference num
            use_qc:  whether or not using QC
        """
        super(US3DDataset, self).__init__()
        self.data_folder = data_folder
        self.mode = mode
        self.view_num = view_num
        self.ref_view = ref_view
        self.use_qc = use_qc

        self.use_fixed_height_range = False
        self.fixed_height_range = None

        # Only four modes
        assert self.mode in ["train", "val", "test", "pred"]
        self.sample_list = self.build_list()

        self.sample_num = len(self.sample_list)

    def build_list(self):
        # Prepare all training samples
        # if self.mode == "pred":
        #     sample_list = gen_all_mvs_list_rpc(self.data_folder, self.view_num)
        # elif self.ref_view < 0:
        #     sample_list = gen_all_mvs_list_rpc(self.data_folder, self.view_num)
        # else:
        #     sample_list = gen_ref_list_rpc(self.data_folder, self.view_num, self.ref_view)
        sample_list = gen_imgs(self.data_folder,self.view_num, self.ref_view)
        if self.mode == "pred":
            sample_list = sample_list
        elif self.mode == "train":
            sample_list = sample_list
        elif self.mode == "test":
            total = len(sample_list)
            keep = int(total * 0.3)
            sample_list = random.sample(sample_list, keep)
        elif self.mode == "val":
            total = len(sample_list)
            keep = int(total * 0.3)
            sample_list = sample_list[:keep]

        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def get_sample(self, idx):
        imgs = self.sample_list[idx]
        rpcs = [os.path.splitext(img)[0] + ".rpc" for img in imgs]
        height_paths = []
        for img_path in imgs:
            img_name = os.path.basename(img_path).replace('.tif', '_heightmap.tif')
            height_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'heightmap2', img_name)
            height_paths.append(height_path)

        sematic_paths = []
        for img_path in imgs:
            img_name = os.path.basename(img_path).replace('.tif', '_sematicmap.tif')
            sematic_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'sematicmap', img_name)
            sematic_paths.append(sematic_path)

        centered_images = []
        rpc_paramters = []

        # Height
        with rasterio.open(height_paths[self.ref_view - 1]) as src:
            depth_image = src.read(1).astype(np.float32)

        with rasterio.open(sematic_paths[self.ref_view - 1]) as src:
            sematic_image = src.read(1).astype(np.float32)
        # ================== 新增 depth_values 决策逻辑 ==================
        # 判断是否使用固定height range
        if hasattr(self, "use_fixed_height_range") and self.use_fixed_height_range and hasattr(self,
                                                                                               "fixed_height_range"):
            depth_min, depth_max = self.fixed_height_range
            # print(f"[Info] Using fixed height range: {depth_min} ~ {depth_max}")
        else:
            _, depth_max, depth_min = load_rpc_as_array(rpcs[self.ref_view - 1])
            # print(f"[Info] Using RPC-derived height range: {depth_min} ~ {depth_max}")

        depth_values = np.array([depth_min, depth_max], dtype=np.float32)
        # ================================================================

        for view in range(self.view_num):
            # Images
            if self.mode == "train":
                image = image_augment(read_img(imgs[view]))
            else:
                image = read_img(imgs[view])
            image = np.asarray(image)

            # RPC
            rpc, _, _ = load_rpc_as_array(rpcs[view])
            rpc_paramters.append(rpc)
            centered_images.append(center_image(image))

        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2])
        rpc_paramters = np.array(rpc_paramters)

        # Mask
        mask = np.float32((depth_image >= depth_min) * 1.0) * np.float32((depth_image <= depth_max) * 1.0)

        h, w = depth_image.shape
        depth_ms = {
            "stage1": cv2.resize(depth_image, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_image, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": depth_image
        }
        mask_ms = {
            "stage1": cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": mask
        }

        # Multi-stage RPC parameters
        stage2_rpc = rpc_paramters.copy()
        stage2_rpc[:, 0] /= 2
        stage2_rpc[:, 1] /= 2
        stage2_rpc[:, 5] /= 2
        stage2_rpc[:, 6] /= 2

        stage3_rpc = rpc_paramters.copy()
        stage3_rpc[:, 0] /= 4
        stage3_rpc[:, 1] /= 4
        stage3_rpc[:, 5] /= 4
        stage3_rpc[:, 6] /= 4

        rpc_paramters_ms = {
            "stage1": stage3_rpc,
            "stage2": stage2_rpc,
            "stage3": rpc_paramters
        }

        path = imgs[0]
        out_view = os.path.basename(os.path.dirname(os.path.dirname(path)))
        out_name = os.path.splitext(os.path.basename(path))[0]

        return {"imgs": centered_images,
                "cam_para": rpc_paramters_ms,
                "depth": depth_ms,
                "sematic":sematic_image,
                "mask": mask_ms,
                "depth_values": depth_values,
                "out_view": out_view,
                "out_name": out_name
                }

    def get_pred_sample(self, idx):
        imgs = self.sample_list[idx]
        rpcs = [os.path.splitext(img)[0] + ".rpc" for img in imgs]

        centered_images = []
        rpc_paramters = []

        if hasattr(self, "use_fixed_height_range") and self.use_fixed_height_range and hasattr(self,
                                                                                               "fixed_height_range"):
            depth_min, depth_max = self.fixed_height_range
            print(f"[Info] Using fixed height range for prediction: {depth_min} ~ {depth_max}")
        else:
            rpc = load_rpc_as_array(rpcs[self.ref_view - 1])[0]
            depth_max = rpc[5] + rpc[6]
            depth_min = rpc[5] - rpc[6]
            print(f"[Info] Using RPC-derived height range for prediction: {depth_min} ~ {depth_max}")

        depth_values = np.array([depth_min, depth_max], dtype=np.float32)
        ## ===================================================

        for view in range(self.view_num):
            image = read_img(imgs[view])
            image = np.asarray(image)

            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1)
            elif image.ndim == 3 and image.shape[2] in [1, 3]:
                pass  # 正常
            else:
                raise ValueError(f"[Error] Unexpected image shape: {image.shape}")

            rpc, _, _ = load_rpc_as_array(rpcs[view])
            rpc_paramters.append(rpc)
            centered_images.append(center_image(image))

        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2])
        rpc_paramters = np.array(rpc_paramters)

        # Multi-stage RPC parameters
        stage2_rpc = rpc_paramters.copy()
        stage2_rpc[:, 0] /= 2
        stage2_rpc[:, 1] /= 2
        stage2_rpc[:, 5] /= 2
        stage2_rpc[:, 6] /= 2

        stage3_rpc = rpc_paramters.copy()
        stage3_rpc[:, 0] /= 4
        stage3_rpc[:, 1] /= 4
        stage3_rpc[:, 5] /= 4
        stage3_rpc[:, 6] /= 4

        rpc_paramters_ms = {
            "stage1": stage3_rpc,
            "stage2": stage2_rpc,
            "stage3": rpc_paramters
        }

        path = imgs[0]
        out_view = os.path.basename(os.path.dirname(os.path.dirname(path)))
        out_name = os.path.splitext(os.path.basename(path))[0]

        return {"imgs": centered_images,
                "cam_para": rpc_paramters_ms,
                "depth_values": depth_values,
                "out_view": out_view,
                "out_name": out_name
                }

    def get_sample_qc(self, idx):
        imgs = self.sample_list[idx]
        rpcs = [os.path.splitext(img)[0] + ".rpc" for img in imgs]
        height_paths = []
        for img_path in imgs:
            img_name = os.path.basename(img_path).replace('.tif', '_heightmap.tif')
            height_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'heightmap2', img_name)
            height_paths.append(height_path)

        # semantic_paths = []
        # for img_path in imgs:
        #     img_name = os.path.basename(img_path).replace('.tif', '_semanticmap.tif')
        #     semantic_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'semanticmap', img_name)
        #     semantic_paths.append(semantic_path)

        centered_images = []
        rpc_paramters = []

        # Height
        with rasterio.open(height_paths[self.ref_view - 1]) as src:
            depth_image = src.read(1).astype(np.float32)

        # ================== 新增 depth_values 决策逻辑 ==================
        if hasattr(self, "use_fixed_height_range") and self.use_fixed_height_range and hasattr(self,
                                                                                               "fixed_height_range"):
            depth_min, depth_max = self.fixed_height_range
            print(f"[Info] Using fixed height range (QC) for sample: {depth_min} ~ {depth_max}")
        else:
            rpc = load_rpc_as_qc_tensor(rpcs[self.ref_view - 1])
            depth_max = rpc["height_off"] + rpc["height_scale"]
            depth_min = rpc["height_off"] - rpc["height_scale"]
            print(f"[Info] Using RPC-derived height range (QC) for sample: {depth_min} ~ {depth_max}")
        depth_values = np.array([depth_min, depth_max], dtype=np.float32)
        # ================================================================

        for view in range(self.view_num):
            # Images
            if self.mode == "train":
                image = image_augment(read_img(imgs[view]))
            else:
                image = read_img(imgs[view])
            image = np.array(image)

            # Cameras
            rpc = load_rpc_as_qc_tensor(rpcs[view])
            rpc_paramters.append(rpc)
            centered_images.append(center_image(image))

        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2])

        # Mask
        mask = np.float32((depth_image >= depth_min) * 1.0) * np.float32((depth_image <= depth_max) * 1.0)

        h, w = depth_image.shape
        depth_ms = {
            "stage1": cv2.resize(depth_image, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_image, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": depth_image
        }
        mask_ms = {
            "stage1": cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": mask
        }

        # Multi-stage RPC parameters (QC form)
        stage2_rpc = copy.deepcopy(rpc_paramters)
        for v in range(len(rpc_paramters)):
            stage2_rpc[v]["line_off"] /= 2
            stage2_rpc[v]["samp_off"] /= 2
            stage2_rpc[v]["line_scale"] /= 2
            stage2_rpc[v]["samp_scale"] /= 2

        stage3_rpc = copy.deepcopy(rpc_paramters)
        for v in range(len(rpc_paramters)):
            stage3_rpc[v]["line_off"] /= 4
            stage3_rpc[v]["samp_off"] /= 4
            stage3_rpc[v]["line_scale"] /= 4
            stage3_rpc[v]["samp_scale"] /= 4

        rpc_paramters_ms = {
            "stage1": stage3_rpc,
            "stage2": stage2_rpc,
            "stage3": rpc_paramters
        }

        path = imgs[0]
        out_view = os.path.basename(os.path.dirname(os.path.dirname(path)))
        out_name = os.path.splitext(os.path.basename(path))[0]

        return {"imgs": centered_images,
                "cam_para": rpc_paramters_ms,
                "depth": depth_ms,
                "mask": mask_ms,
                "depth_values": depth_values,
                "out_view": out_view,
                "out_name": out_name
                }

    def get_pred_sample_qc(self, idx):
        imgs = self.sample_list[idx]
        rpcs = [os.path.splitext(img)[0] + ".rpc" for img in imgs]

        centered_images = []
        rpc_paramters = []

        if hasattr(self, "use_fixed_height_range") and self.use_fixed_height_range and hasattr(self,
                                                                                               "fixed_height_range"):
            depth_min, depth_max = self.fixed_height_range
            print(f"[Info] Using fixed height range (QC Prediction): {depth_min} ~ {depth_max}")
        else:
            rpc = load_rpc_as_qc_tensor(rpcs[self.ref_view - 1])
            depth_max = rpc["height_off"] + rpc["height_scale"]
            depth_min = rpc["height_off"] - rpc["height_scale"]
            print(f"[Info] Using RPC-derived height range (QC Prediction): {depth_min} ~ {depth_max}")

        depth_values = np.array([depth_min, depth_max], dtype=np.float32)
        ## ===================================================

        for view in range(self.view_num):
            image = read_img(imgs[view])
            image = np.asarray(image)
            image = np.expand_dims(image, axis=-1)

            rpc = load_rpc_as_qc_tensor(rpcs[view])
            rpc_paramters.append(rpc)
            centered_images.append(center_image(image))

        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2])

        # Multi-stage RPC parameters (QC form)
        stage2_rpc = copy.deepcopy(rpc_paramters)
        for v in range(len(rpc_paramters)):
            stage2_rpc[v]["line_off"] /= 2
            stage2_rpc[v]["samp_off"] /= 2
            stage2_rpc[v]["line_scale"] /= 2
            stage2_rpc[v]["samp_scale"] /= 2

        stage3_rpc = copy.deepcopy(rpc_paramters)
        for v in range(len(rpc_paramters)):
            stage3_rpc[v]["line_off"] /= 4
            stage3_rpc[v]["samp_off"] /= 4
            stage3_rpc[v]["line_scale"] /= 4
            stage3_rpc[v]["samp_scale"] /= 4

        rpc_paramters_ms = {
            "stage1": stage3_rpc,
            "stage2": stage2_rpc,
            "stage3": rpc_paramters
        }

        path = imgs[0]
        out_view = os.path.basename(os.path.dirname(os.path.dirname(path)))
        out_name = os.path.splitext(os.path.basename(path))[0]

        return {"imgs": centered_images,
                "cam_para": rpc_paramters_ms,
                "depth_values": depth_values,
                "out_view": out_view,
                "out_name": out_name
                }

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        if self.mode != "pred":
            if self.use_qc:
                return self.get_sample_qc(idx)
            else:
                return self.get_sample(idx)
        else:
            if self.use_qc:
                return self.get_pred_sample_qc(idx)
            else:
                return self.get_pred_sample(idx)


def gen_imgs(input_folder, view_num, ref_view=1):
    """
    灵活读取影像组合：
    - 如果 input_folder 下是区域（JAX、OMA等），递归所有区域；
    - 如果 input_folder 是单个区域（直接含子块如144,179），只处理当前区域。
    每个组合是一个 list，整体返回 list[list[str]]。
    """
    all_combinations = []

    # 判断 input_folder 是总目录还是区域目录
    has_subdirs = any(os.path.isdir(os.path.join(input_folder, f)) for f in os.listdir(input_folder))

    # 判断是区域还是更上一级
    possible_subfolder = os.path.join(input_folder, os.listdir(input_folder)[0])
    if os.path.isdir(possible_subfolder) and 'image' in os.listdir(possible_subfolder):
        # 当前是区域目录，直接处理
        region_folders = [input_folder]
    else:
        # 当前是总目录，遍历多个区域
        region_folders = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if
                          os.path.isdir(os.path.join(input_folder, f))]

    for region_folder in region_folders:
        subfolders = [f for f in os.listdir(region_folder) if os.path.isdir(os.path.join(region_folder, f))]

        for subfolder in subfolders:
            image_folder_path = os.path.join(region_folder, subfolder, 'image')

            if not os.path.isdir(image_folder_path):
                continue

            json_files = [f for f in os.listdir(image_folder_path) if re.match(r"selected(_\d+)?\.json$", f)]
            if not json_files:
                print(f"[!] Warning: No selected_*.json found in {image_folder_path}, skipping...")
                continue

            for json_file in json_files:
                json_path = os.path.join(image_folder_path, json_file)
                try:
                    with open(json_path, 'r') as f:
                        selected = json.load(f).get("selected_images", [])
                    if len(selected) >= view_num:
                        selected_paths = [os.path.join(image_folder_path, fname) for fname in selected]
                        all_combinations.append(selected_paths[:view_num])
                except Exception as e:
                    print(f"[x] Error reading {json_path}: {e}")
                    continue

    return all_combinations

def load_rpc_from_image(image_path):
    # 打开影像
    ds = gdal.Open(image_path)

    if ds is None:
        raise Exception(f"无法打开影像文件: {image_path}")

    # 获取影像的元数据
    metadata = ds.GetMetadata()

    # 查找 RPC 信息
    rpc_info = {}
    for key, value in metadata.items():
        if key.startswith("RPC_"):
            rpc_info[key] = value

    if len(rpc_info) == 0:
        raise Exception("没有找到RPC信息!")

    # 打印 RPC 信息（可选）
    for key, value in rpc_info.items():
        print(f"{key}: {value}")

    # 你可以根据需要返回 RPC 信息的详细数据
    return rpc_info



if __name__ == "__main__":
    # gen_imgs(r"E:\Data\US3D\US3D-MVS\JAX",3,1)
    dataset = US3DDataset(r"H:\MVS-Dataset\Test\JAX","test",3,1,True)
    print("OK")
    print(dataset)
    print("OK")