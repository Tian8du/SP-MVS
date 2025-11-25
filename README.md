# SP-MVS

**SP-MVS: A Structure-Preserving Network for Edge-Aware Height Estimation from Multi-View Satellite Images**
> 🛰️ **Publication Notice**  
> This work has been **accepted for publication** in the  
> *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTARS)*,  
> titled:  
> **“A Structure-Preserving Multi-View Stereo Network for Edge-Aware Height Estimation from Satellite Images”**.  
>  
> The paper appears as a **Regular Paper** in an upcoming JSTARS issue.  
> All datasets (US3D, WHU-TLC, MVS3D) and the full source code have been publicly released.

---




The US3D dataset and the full codes will be released after publication.

### 1. Download the datasets (WHU-TLC, US3D-MVS, DTU and MVS3D)

To evaluate/train this method, you will need to download the required datasets and corresponding model weights. 
* [WHU-TLC](https://github.com/WHU-GPCV/SatMVS/blob/main/WHU_TLC/readme.md)
* [US3D-MVS](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019)
* [DTU (training data)](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) | [DTU (Depths raw)](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) | [DTU (testing data)](https://drive.google.com/file/d/1rX0EXlUL4prRxrRu2DgLJv2j7-tpUD4D/view?usp=sharing)
* [Model SP-MVS](https://drive.google.com/file/d/1b8i1u69_9yMPJyqGcuTkCocyg0rVg4P3/view?usp=sharing).

US3D-MVS and MVS3D can be acquired by my MVS tool  [Sat-MVS-Dataset](https://github.com/Tian8du/Sat-MVS-Dataset).

By default `dataset_root` in `train.py`, we will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the `data` folder

```Shell
├── data
    ├── WHU-TLC
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
    ├── US3D-MVS
        ├── train_OMA
            ├── image
            ├── height_map
            ├── rpcs
        ├── test_JAX
            ├── image
            ├── height_map
            ├── rpcs
    ├── MVS3D
        ├── train_MVS3D
            ├── image
            ├── height_map
            ├── rpcs
        ├── test_MVS3D
            ├── image
            ├── height_map
            ├── rpcs
```

Notes: the tools for making US3D and MVS3D will be provided later after paper is published.
### 2. Create and activate conda environment
conda create -n MVS python=3.7 -y
conda activate MVS

### Install PyTorch (CUDA 11.0)
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

### Install common dependencies
pip install gdal==2.4.2.* \
            matplotlib \
            opencv-python \
            imageio \
            tensorboardX


### 3. Train
for example:Train on WHU-TLC dataset using RPC warping:

`python train.py --mode="train" --model="casmvs" --geo_model="rpc" --dataset_root=[Your dataset root] --batch_size=1 --min_interval=[GSD(resolution of the image)] --gpu_id="0"`


### 4. Predict

`python predict.py --model="casmvs" --geo_model="rpc" --dataset_root=[Your dataset] --loadckpt=[A checkpoint]`


## Acknowledgement
This project is heavily inspired by the paper “Surface Depth Estimation From Multiview Stereo Satellite Images With Distribution Contrast Network.” Another significant breakthrough that has greatly influenced this work is SatMVSF developed by the GPCV group at Wuhan University.

A substantial portion of the code in this repository is adapted from the implementations described in their work. I sincerely appreciate the authors’ valuable contributions to the field, which have significantly facilitated and accelerated my research.

### Citation
If you find this repository helpful in your research, please consider citing our paper:

```bibtex
@ARTICLE{11218776,
  author={Liu, Chen and Jiang, Yonghua and Wang, Dong and Shen, Xin and Wang, Yunming and Zhang, Guangbin},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={A Structure-Preserving Multiview Stereo Network for Edge-Aware Height Estimation From Satellite Images}, 
  year={2025},
  volume={18},
  number={},
  pages={28782-28796},
  keywords={Satellite images;Three-dimensional displays;Feature extraction;Computational modeling;Accuracy;Image edge detection;Transformers;Estimation;Costs;Satellite broadcasting;3-D reconstruction;deep learning;multiview stereo (MVS);remote sensing},
  doi={10.1109/JSTARS.2025.3626009}}
