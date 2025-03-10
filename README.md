# TBFE-Network

Triple-Branch Feature Extraction Network for Height Estimation from Multiview Stereo Satellite Images. You Can follow the following steps to train and predict TBFE-Net on WHU-TLC

## download the WHU-TLC dataset

You can get the WHU-TLC dataset from **GPCV**, the link is (http://gpcv.whu.edu.cn/)
To evaluate/train this method, you will need to download the required datasets. 
* [WHU-TLC](https://github.com/WHU-GPCV/SatMVS/blob/main/WHU_TLC/readme.md) Please rename the "open_dataset" to "open_dataset_rpc".
* [DTU (training data)](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view)
* [DTU (Depths raw)](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip)
* [DTU (testing data)](https://drive.google.com/file/d/1rX0EXlUL4prRxrRu2DgLJv2j7-tpUD4D/view?usp=sharing)
* [Model (TBFE-UCS)](https://drive.google.com/file/d/1b8i1u69_9yMPJyqGcuTkCocyg0rVg4P3/view?usp=sharing)

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
```

## Environment Preparation
```Shell
conda create -n MVS python=3.7
conda activate MVS
wget -c https://www.sqlite.org/2021/sqlite-autoconf-3340100.tar.gz
tar -xvf sqlite-autoconf-3340100.tar.gz
cd sqlite-autoconf-3340100
vim sqlite3.c
```
Add macros under "include"
```Shell
#define SQLITE_CORE 1
#define SQLITE_AMALGAMATION 1
#ifndef SQLITE_PRIVATE
# define SQLITE_PRIVATE static
#endif
#define SQLITE_ENABLE_COLUMN_METADATA 1        //Please pay attention to this line
 
/************** Begin file ctime.c *******************************************/
/*
```
recompile
```Shell
./configure
make
sudo make uninstall
sudo make install
#Verify that the installation was successful
sqlite3 --version
```
Download proj (version 6.3.2) source code, unzip it and compile and install it
```Shell
proj-5.2.0
wget https://download.osgeo.org/proj/proj-6.3.2.tar.gz
tar -zxvf proj-6.3.2.tar.gz
#Go to the directory and compile
cd proj-6.3.2
./configure
make
make install
ldconfig
proj --version
```
Download geos (version 3.8.1), unzip it, compile and install it.
```Shell
wget http://download.osgeo.org/geos/geos-3.8.1.tar.bz2
tar -jxvf geos-3.8.1.tar.bz2
cd geos-3.8.1
./configure
make
make install 
ldconfig
geos-config --version
```
gdal(2.4.2)
```Shell
pip install setuptools==57.5.0
sudo add-apt-repository ppa:ubuntugis && sudo apt update
sudo apt install gdal-bin
gdalinfo --version  # 假设输出为2.4.2
```
```Shell
pip install gdal==2.4.2.*
```
or
```Shell
wget -c http://download.osgeo.org/gdal/2.4.2/gdal-2.4.2.tar.gz
tar -zxvf gdal-2.4.2.tar.gz
cd /gdal-2.4.2/swig/python/
python setup.py build
python setup.py install
```
```Shell
python
from osgeo import gdal
```
```Shell
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorboardX
pip install matplotlib
pip install opencv-python
pip install imageio
```

## Train

Train on WHU-TLC dataset using RPC warping:

`python train.py --mode="train" --model="eucs" --geo_model="rpc" --dataset_root=[Your dataset root] --batch_size=1 --min_interval=[GSD(resolution of the image)] --gpu_id="0"`

Train on WHU-TLC dataset using homography warping:

`python train.py --mode="train" --model="eucs" --geo_model="pinhole" --dataset_root=[Your dataset root] --batch_size=1 --min_interval=[GSD(resolution of the image)] --gpu_id="0"`

### Predict
If you want to predict your own dataset, you need to If you want to predict on your own dataset, you need to first organize your dataset into a folder similar to the WHU-TLC dataset. And then run:

`python predict.py --model="red" --geo_model="rpc" --dataset_root=[Your dataset] --loadckpt=[A checkpoint]`


### Acknowledgement
This project is heavily inspired by the paper "Surface Depth Estimation From Multiview Stereo Satellite Images With Distribution Contrast Network". A significant portion of the code in this repository is derived from the implementations described in this work. I sincerely appreciate the authors' contributions to the field, which have greatly facilitated my research.

If you find this work useful, please consider citing the original paper.


