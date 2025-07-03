# SP-MVS

**SP-MVS: A Structure-Preserving Network for Edge-Aware Height Estimation from Multi-View Satellite Images**

## download the WHU-TLC dataset

You can get the WHU-TLC dataset from [**GPCV**](http://gpcv.whu.edu.cn/).
To evaluate/train this method, you will need to download the required datasets and corresponding model weights. 
* [WHU-TLC](https://github.com/WHU-GPCV/SatMVS/blob/main/WHU_TLC/readme.md) Please rename the "open_dataset" to "open_dataset_rpc".
* [US3D-MVS](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019)
* [DTU (training data)](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view)
* [DTU (Depths raw)](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip)
* [DTU (testing data)](https://drive.google.com/file/d/1rX0EXlUL4prRxrRu2DgLJv2j7-tpUD4D/view?usp=sharing)
* [Model SPMVS](https://drive.google.com/file/d/1b8i1u69_9yMPJyqGcuTkCocyg0rVg4P3/view?usp=sharing).

By default `dataset_root` in `train.py`, we will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the `data` folder

```Shell
├── data
    ├── TLC
        ├── Open
        ├── open_dataset_pinhole
        ├── open_dataset_rpc
    ├── DTU
        ├── mvs_training
            ├── Cameras
            ├── Depths
            ├── Depths_raw
            ├── Rectified
        ├── dtu_test
    ├── US3D
        ├── train_OMA
            ├── image
            ├── height_map
            ├── rpcs
        ├── test_JAX
            ├── image
            ├── height_map
            ├── rpcs
    ├── MVS3D
        ├── train
            ├── image
            ├── height_map
            ├── rpcs
        ├── test
            ├── image
            ├── height_map
            ├── rpcs
```

# Create and activate conda environment
conda create -n MVS python=3.7 -y
conda activate MVS

# Install PyTorch (CUDA 11.0)
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# Install common dependencies
pip install gdal==2.4.2.* \
            matplotlib \
            opencv-python \
            imageio \
            tensorboardX


# Train

Train on WHU-TLC dataset using RPC warping:

`python train.py --mode="train" --model="eucs" --geo_model="rpc" --dataset_root=[Your dataset root] --batch_size=1 --min_interval=[GSD(resolution of the image)] --gpu_id="0"`


# Predict
 If you want to predict on your own dataset, you need to first organize your dataset into a folder similar to the WHU-TLC dataset. And then run:

`python predict.py --model="red" --geo_model="rpc" --dataset_root=[Your dataset] --loadckpt=[A checkpoint]`


### Acknowledgement
This project is heavily inspired by the paper "Surface Depth Estimation From Multiview Stereo Satellite Images With Distribution Contrast Network". A significant portion of the code in this repository is derived from the implementations described in this work. I sincerely appreciate the authors' contributions to the field, which have greatly facilitated my research.

If you find this work useful, please consider citing the original paper.


