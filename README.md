[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) ![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/github/layumi/3D-Magic-Mirror.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/github/layumi/3D-Magic-Mirror/context:python)

## 3D Magic Mirror: Clothing Reconstruction from a Single Image via a Causal Perspective
![](./doc/camera_github.gif)

![](./doc/rainbow_github.gif)

Cub Bird         |  Market-HQ Exchange
:-------------------------:|:-------------------------:
![](https://github.com/layumi/3D-Magic-Mirror/blob/master/doc/cub.gif?raw=true)  |  ![](https://github.com/layumi/3D-Magic-Mirror/blob/master/doc/current_rainbow_github.gif?raw=true)


[[Project]](https://www.zdzheng.xyz/publication/3D-Magic2022) [[Code]](https://github.com/layumi/3D-Magic-Mirror) [[Paper]](https://zdzheng.xyz/files/3D_Recon.pdf)

3D Magic Mirror: Clothing Reconstruction from a Single Image via a Causal Perspective, arXiv preprint arXiv:2204.13096, 2022.<br>[Zhedong Zheng](http://zdzheng.xyz/), [Jiayin Zhu](https://github.com/viridityzhu), [Wei Ji](https://jiwei0523.github.io/), [Yi Yang](https://www.uts.edu.au/staff/yi.yang), [Tat-Seng Chua](https://www.chuatatseng.com/)

<meta name="og:image" content="https://github.com/layumi/3D-Magic-Mirror/blob/master/doc/current_rainbow_github.gif?raw=true">

## Table of contents
* [News](#news)
* [Features](#features)
* [Prerequisites](#prerequisites)
* [Getting Started](#getting-started)
    * [Installation](#installation)
    * [Dataset Preparation](#dataset-preparation)
    * [Testing](#testing)
    * [Training](#training)
* [Citation](#citation)
* [Related Work](#related-work)
* [License](#license)

## News
- We will release a new arXiv recently.  We fix the stability problem. The results in paper are averaged with at least three runs. 

## Features
We have supported:

- Train and test on 3 datasets: CUB, ATR, Market
- Generate all figures in the paper 

## Prerequisites

- Linux
- Python >= 3.7
- CUDA >= 11 (with `nvcc` installed)

If you use CUDA 10, please download the corresponding pytorch and kaolin 0.9 to match.  

## Getting Started
### Installation

- Clone this repo:
```sh
$ git clone https://github.com/layumi/3D-Magic-Mirror.git
$ cd 3D-Magic-Mirror/
```

- Install requirements

```sh
$ conda create --name magic --file spec-file.txt
$ conda activate magic
$ pip install pytorch_msssim
```

* gcc is needed by kaolin. If you have gcc about 7.3.0 - 9.5.0 (latest gcc may not work as well), please skip this step. 
(Update your gcc as follows if your gcc is too old (<=7.3) and you do not have sudo rights.)  

```sh
$ conda config --add channels conda-forge # add conda forge channel
#$ conda install gcc_linux-64=9.4.0 gcc_impl_linux-64=9.4.0  gxx_linux-64=9.4.0 gxx_impl_linux-64=9.4.0 # I have included in spec-file.txt
$ ln x86_64-conda-linux-gnu-gcc gcc # cd your_anaconda/envs/magic/bin
$ ln x86_64-conda-linux-gnu-g++ g++ 
```

* install Kaolin Library
(You need to mute/comment some warning like cython to install it.)
```sh
$ git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
$ git checkout v0.12.0
$ python setup.py develop
```

* Others: tqdm, trimesh, imageio, etc.

Our code is tested on PyTorch 1.9.0+ and torchvision 0.10.0+.

### Dataset Preparation


Download tool:
```
$ pip install gdown 
$ pip install --upgrade gdown #!!important!!
```

OR 

Install gdrive for fast download the dataset from Google Driver. It is good for all command line users. (https://github.com/prasmussen/gdrive/releases/tag/2.1.1 )

```
wget https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_2.1.1_linux_386.tar.gz
tar -xzvf gdrive_2.1.1_linux_386.tar.gz
```


Download the dataset:

- [Market-HQ](https://github.com/layumi/HQ-Market)

Download the processed data from [Google Drive](https://drive.google.com/file/d/10gNi3n8Iny4O4MOZRs5dNFICfj8ri9eW/view?usp=sharing). Or use the gdrive tool to download:

```
gdown https://drive.google.com/uc?id=10gNi3n8Iny4O4MOZRs5dNFICfj8ri9eW
OR 
gdrive download 10gNi3n8Iny4O4MOZRs5dNFICfj8ri9eW
```

- Market-Seg-HMR (We deploy smoothed HMR results as foreground marks. We found human parsing results are sensitive to input domain, such as illumination.)

Download the processed data from [Google Drive](https://drive.google.com/file/d/1JwQTQA4F9WOqLAc7fhQN0DTYwnv6XrAR/view?usp=sharing). Or use the gdrive tool to download:

```
gdown https://drive.google.com/uc?id=1JwQTQA4F9WOqLAc7fhQN0DTYwnv6XrAR
OR 
gdrive download 1JwQTQA4F9WOqLAc7fhQN0DTYwnv6XrAR
```

- CUB

Download the processed data from [Google Drive](https://drive.google.com/file/d/1SkX_FWUfLOaTr371TBkQnDH9oDJ5Khwc/view?usp=sharing). Or use the gdrive tool to download:

```
gdown https://drive.google.com/uc?id=1SkX_FWUfLOaTr371TBkQnDH9oDJ5Khwc
OR 
gdrive download 1SkX_FWUfLOaTr371TBkQnDH9oDJ5Khwc
```

- ATR

Download the processed data from [Google Drive](https://drive.google.com/file/d/1kpsMDrbM4FQqtP7Y1nKslp4OlRKNvbaL/view). Or use the gdrive tool to download:

```
gdown https://drive.google.com/uc?id=1kpsMDrbM4FQqtP7Y1nKslp4OlRKNvbaL
OR
gdrive download 1kpsMDrbM4FQqtP7Y1nKslp4OlRKNvbaL
```



### Preparation: 

Before prepare, the folder is like:
```
├── 3D-Magic-Mirror/
|   |-- data/
|       |-- CUB_Data/
├── kaolin/
├── Market/
│   ├── hq/
|       |-- seg_hmr/
|-- ATR/
|   |-- humanparsing/
```

Only Market dataset requires preparation, and other datasets are ready to run after download.  
This code will calculate the ratio of foreground against background. 
During training, we will drop few wrong masks or ill-detected person.

Note to modify the dataset path to your own path.

```bash
python prepare_market.py         
python prepare_ATR.py
python prepare_cub.py
```


### Testing

#### Download the trained model

- Trained model 

You may download it from [GoogleDrive-Market](https://drive.google.com/file/d/1-eqnnFt3D7-jUelJ5uj_4QcFv-TMok8c/view?usp=sharing), [GoogleDrive-CUB](https://drive.google.com/file/d/1urxUeaULn2DNM-4XAcZ2_OmtGJuCSeYO/view?usp=sharing) and move it to the `log`.
Or directly use the following code:
```
gdown 1-eqnnFt3D7-jUelJ5uj_4QcFv-TMok8c #Market model
gdown 1urxUeaULn2DNM-4XAcZ2_OmtGJuCSeYO #CUB model
```

```
├── log/
│   ├── CamN2_MKT_wgan_b48_lr0.5_em7_update-1_lpl_reg0.1_data2_m2_flat20_depthR0.15_drop220_gap2_beta0.95_clean67/
|       |-- ckpts/
│   ├──CUB_wgan_b48_ic1_hard_bg_L1_ganW0_lr0.7_em7_update-1_chf_lpl_reg0.1_data2_depthC0.1_flat10_drop220_gap2_beta0.95_bn_restart1_contour0.1/
|       |-- ckpts/
```

- Visualization 
```bash
python show_rainbow2.py --name CamN2_MKT_wgan_b48_lr0.5_em7_update-1_lpl_reg0.1_data2_m2_flat20_depthR0.15_drop220_gap2_beta0.95_clean67
```
It will generate the five gif animations in the `log/your_model_name/`.
(We manually select some hard index to show the result.)

`current_rainbow.gif`: Swapping appearnce. 

`current_rotation.gif`: Rotation via azumith.

`current_rotation_ele.gif`: Rotation via elevation. 

`current_rotation_dist.gif`: Change distance to the camera. 

`current_rotation_XY.gif`: Shift the camera in X-axis and Y-axis. 

- Test Flops, maskIoU and SSIM 
```bash
python test.py --name CamN2_MKT_wgan_b48_lr0.5_em7_update-1_lpl_reg0.1_data2_m2_flat20_depthR0.15_drop220_gap2_beta0.95_clean67
or
python test.py --name ATR2_wgan_b48_ganW0_lr0.55_em7_update-1_chf_lpl_reg0.1_m2_recon2_flat10_depthR0.15_data2_drop222_gap2_beta0.95_s96_clean1826 
or 
python test.py --name CUB_wgan_b48_ic1_hard_bg_L1_ganW0_lr0.7_em7_update-1_chf_lpl_reg0.1_data2_depthC0.1_flat10_drop220_gap2_beta0.95_bn_restart1_contour0.1 
```
**Please make sure the dataset name in your model. We use model name to set the test dataset.**

### Training

- Training on Market (64*128)
```sh
python train_market.py --name CamN2_MKT_wgan_b48_lr0.5_em7_update-1_lpl_reg0.1_data2_m2_flat20_depthR0.15_drop220_gap2_beta0.95_clean67  --clean 0.36,0.49  --imageSize 64 --batch 48 --gan_type wgan --bg --L1 --ganw 0 --hard --lr 5e-5 --em 7 --update_shape -1  --lambda_data 2 --unmask 2  --lambda_flat 0.02 --lambda_depthR 0.15  --drop 0.2,0.2,0  --em_gap 2 --beta1 0.95   --pretrainc none
```

- Training on CUB (128*128)
```sh
python train.py --name CUB_wgan_b48_ic1_hard_bg_L1_ganW0_lr0.7_em7_update-1_chf_lpl_reg0.1_data2_depthC0.1_flat10_drop220_gap2_beta0.95_bn_restart1_contour0.1  --drop 0.2,0.2,0 --imageSize 128 --batch 48 --gan_type wgan --bg --L1 --ganw 0 --hard --lr 7e-5 --em 7 --update_shape -1  --lambda_data 2 --lambda_depthC 0.1 --lambda_flat 0.01   --unmask 2   --em_gap 2 --beta1 0.95 --update_bn --gamma 0.1 --scheduler restart1 --lambda_contour 0.1
```

- Training on ATR (96*160) **We suggest to run this dataset on A100 or A5000, instead of P5000 or R5000. The result is more stable on A series GPUs. It may be due to the float accuracy.**
```sh
python train_ATR2.py --name ATR2_wgan_b48_ganW0_lr0.55_em7_update-1_chf_lpl_reg0.1_m2_recon2_flat10_depthR0.15_data2_drop222_gap2_beta0.95_s96_clean1826  --imageSize 96 --batch 48 --gan_type wgan --bg --L1 --ganw 0 --hard --lr 5.5e-5 --em 7 --update_shape -1 --unmask 2 --lambda_data 2  --lambda_flat 0.01 --lambda_depthR 0.15  --drop 0.2,0.2,0.2  --em_gap 2 --beta1 0.95 --ratio 1.666666 --clean 0.18,0.26    --pretrainc none  
```

### Illustrations. 
- `em` is for the accumulated template update. If we set smooth=6, the movement will be the neighbor move. If we set smooth=7, the movement will be the smooth over neighbors again. by default 7

- `clip` is to truncate the movement. by default 0.05

- `em_step` is the initial moving speed, which is decay by 0.99 during training. by default 0.1

## Citation

Please cite this paper if it helps your research:

```bibtex
@article{zheng2022magic,
  title={3D Magic Mirror: Clothing Reconstruction from a Single Image via a Causal Perspective},
  author={Zheng, Zhedong and Zhu, Jiayin and Ji, Wei and Yang, Yi and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:2204.13096},
  year={2022}
}
```

## Related Work
We would like to thank to the great projects in [SMR](https://github.com/dvlab-research/SMR) and [UMR](https://github.com/NVlabs/UMR).

The person re-identification part is from [Pytorch re-ID](https://github.com/layumi/Person_reID_baseline_pytorch)

## License
Copyright (C) 2022 NExT++ Lab. All rights reserved. Licensed under the MIT. The code is released for academic research use only. For commercial use, please contact [zhedongzheng@um.edu.mo](zhedongzheng@um.edu.mo).

