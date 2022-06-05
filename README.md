[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) ![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/github/layumi/3D-Magic-Mirror.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/github/layumi/3D-Magic-Mirror/context:python)

## 3D Magic Mirror: Clothing Reconstruction from a Single Image via a Causal Perspective
![](./doc/Figure7.gif)

![](./doc/Figure10.gif)

<pre align="center" >
<center><img src="./doc/Figure8.gif" width="40%" /></center>
</pre>

[[Project]](https://zdzheng.xyz/publication/3D-Ma2022) [[Code]](https://github.com/layumi/3D-Magic-Mirror) [[Paper]](https://zdzheng.xyz/files/3D_Recon.pdf)

3D Magic Mirror: Clothing Reconstruction from a Single Image via a Causal Perspective, arXiv preprint arXiv:2204.13096, 2022.<br>[Zhedong Zheng](http://zdzheng.xyz/), [Jiayin Zhu](https://github.com/viridityzhu), [Wei Ji](https://jiwei0523.github.io/), [Yi Yang](https://www.uts.edu.au/staff/yi.yang), [Tat-Seng Chua](https://www.chuatatseng.com/)

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


## To DO List
- Check the EMA part code.
- Running on a new machine. 

## Features
We have supported:

- Train and test on 3 datasets: CUB, ATR, Market
- Generate all figures in the paper 

## Prerequisites

- Linux
- Python >= 3.7
- CUDA >= 10.0.130 (with `nvcc` installed)

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
$ pip install tensorboard tensorboardx
$ pip install opencv-python imageio trimesh typing-extensions
$ pip install timm prettytable h5py imgaug smplx munkres joblib pycocotools lap plotly pandas
$ pip install ipywidgets keyboard transforms3d chumpy
```

* gcc is needed by kaolin. If you have latest gcc, please skip this step. (Update your gcc if your gcc is too old (<=7.3) and you do not have sudo rights.)  

```sh
$ conda config --add channels conda-forge # add conda forge channel
$ conda install gcc_linux-64=9.4.0 gcc_impl_linux-64=9.4.0
$ conda install gxx_linux-64=9.4.0 gxx_impl_linux-64=9.4.0
$ ln x86_64-conda-linux-gnu-gcc gcc # cd your_anaconda/envs/magic/bin
$ ln x86_64-conda-linux-gnu-g++ g++ 
```

* install Kaolin Library

```sh
$ git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
$ cd kaolin
$ git checkout v0.9.1
$ python setup.py develop
```

* Others: tqdm, trimesh, imageio, etc.

```sh
$ pip install pytorch_msssim
$ conda install scikit-learn
$ conda install imageio
```

Our code is tested on PyTorch 1.9.0+ and torchvision 0.10.0+.

### Dataset Preparation


Download tool:

Install gdrive for fast download the dataset from Google Driver. It is good for all command line users. (https://github.com/prasmussen/gdrive/releases/tag/2.1.1 )

```
wget https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_2.1.1_linux_386.tar.gz
tar -xzvf gdrive_2.1.1_linux_386.tar.gz
```


Download the dataset:

- [Market-HQ](https://github.com/layumi/HQ-Market)

Download the processed data from [Google Drive](https://drive.google.com/file/d/10gNi3n8Iny4O4MOZRs5dNFICfj8ri9eW/view?usp=sharing). Or use the gdrive tool to download:

```
gdrive download 10gNi3n8Iny4O4MOZRs5dNFICfj8ri9eW
```

- Market-Seg-HMR

Download the processed data from [Google Drive](https://drive.google.com/file/d/1JwQTQA4F9WOqLAc7fhQN0DTYwnv6XrAR/view?usp=sharing). Or use the gdrive tool to download:

```
gdrive download 1JwQTQA4F9WOqLAc7fhQN0DTYwnv6XrAR
```

Note that Market-Seg should be put in the subdirectory of Market-HQ. You may follow this to decompress the two dataset:

```sh
mkdir Market
tar -xzvf Market-hq.tar.gz --directory=./Market
tar -xzvf Market-seg.tar.gz --directory=./Market/hq # make sure market-seg is in the subdirectory of market-hq
```


- CUB

Download the processed data from [Google Drive](https://drive.google.com/file/d/1SkX_FWUfLOaTr371TBkQnDH9oDJ5Khwc/view?usp=sharing). Or use the gdrive tool to download:

```
gdrive download 1SkX_FWUfLOaTr371TBkQnDH9oDJ5Khwc
unzip CUB_SMR_Data.zip
```

- ATR

Download the processed data from [Google Drive](https://drive.google.com/file/d/1kpsMDrbM4FQqtP7Y1nKslp4OlRKNvbaL/view). Or use the gdrive tool to download:

```
gdrive download 1kpsMDrbM4FQqtP7Y1nKslp4OlRKNvbaL
unzip humanparsing -d ATR
```



### Preparation: 

Only Market dataset requires preparation, and other datasets are ready to run after download.  
This code will calculate the ratio of foreground against background. 
During training, we will drop few wrong masks or ill-detected person.

Note to modify the dataset path to your own path.

```bash
python prepare_market.py          # only Market is needed.
```


### Testing

#### Download the trained model

- Trained model 

You may download it from [GoogleDrive](https://drive.google.com/file/d/1NUs2MoCo_gUsUXeHg1i6PqyfAxseJmk9/view?usp=sharing) and move it to the `log`.

```sh
gdrive download 1NUs2MoCo_gUsUXeHg1i6PqyfAxseJmk9
unzip magic_mirror_snapshots.zip -d ./log
```

```
├── log/
│   ├── MKT_wgan_b48_lr0.5_em1_update-1_chf_lpl_reg0.1_data2_m2_flat7_depth0.1_drop0.4_gap2_beta0.9/
|       |-- ckpts/
```

- Visualization 
```bash
python show_rainbow2.py --name  MKT_wgan_b48_lr0.5_em1_update-1_chf_lpl_reg0.1_data2_m2_flat7_depth0.1_drop0.4_gap2_beta0.9 \
   --dataroot ./Market/hq/seg # or path/to/your/Market/hq/seg
```
It will generate the five gif animations in the `./rainbow/`.
(We manually select some hard index to show the result.)

`current_rainbow.gif`: Swapping appearnce. 

`current_rotation.gif`: Rotation via azumith.

`current_rotation_ele.gif`: Rotation via elevation. 

`current_rotation_dist.gif`: Change distance to the camera. 

`current_rotation_XY.gif`: Shift the camera in X-axis and Y-axis. 


- Test Flops, maskIoU and SSIM 
```bash
python test.py --name MKT_wgan_b48_lr0.5_em1_update-1_chf_lpl_reg0.1_data2_m2_flat7_depth0.1_drop0.4_gap2_beta0.9  --dataroot ./Market/hq/seg # or path/to/your/Market/hq/seg
# or
python test.py --name rereATR128_wgan_b48_ganW0_lr0.6_em1_update-1_chf_lpl_reg0.1_m2_recon2_flat10_depth0.2_data4_drop0.4_gap2_beta0.9  --dataroot ./ATR/humanparsing/JPEGImages # or path/to/your/humanparsing/JPEGImages
# or 
python test.py --name CUB_wgan_b16_ic1_hard_bg_L1_ganW0_lr0.3_em1_update-1_chf_lpl_reg0.1_data2_depth0.1_flat10_drop0.2_gap2_beta0.9 --dataroot ./CUB_Data # or path/to/your/CUB_Data
```
**Please make sure the dataset name in your model. We use model name to set the test dataset.**

In `test.py`, the latest_template_file is set by default as `/epoch_xxx_template.obj`, where xxx is the latest epoch. There is one thing to note: in the downloaded trained model, the best template is  `/ckpts/best_mesh.obj`. So **if you are testing the downloaded trained model, you need to comment line 206 and uncomment line 207 in `test.py`** to change the latest_template_file.

### Training
- Training on CUB

```sh
python train.py --name CUB_wgan_b16_ic1_hard_bg_L1_ganW0_lr0.3_em1_update-1_chf_lpl_reg0.1_data2_depth0.1_flat10_drop0.2_gap2_beta0.9 \
                --drop 0.2 \
                --imageSize 128 \
                --batch 16 \
                --gan_type wgan \
                --bg \
                --L1 \
                --ganw 0 \
                --hard \
                --lr 3e-5 \
                --em 1 \
                --update -1 \
                --chamfer \
                --lambda_data 2 \
                --lambda_depth 0.1 \
                --lambda_flat 0.01 \
                --unmask 2 \
                --amsgrad \
                --em_gap 2 \
                --beta1 0.9 \
                --dataroot ./CUB_Data # or path/to/your/CUB_Data
```

- Training on ATR

```sh
python train_ATR.py --name rereATR128_wgan_b48_ganW0_lr0.6_em1_update-1_chf_lpl_reg0.1_m2_recon2_flat10_depth0.2_data4_drop0.4_gap2_beta0.9 \
                    --imageSize 128 \
                    --batch 48 \
                    --gan_type wgan \
                    --bg \
                    --L1 \
                    --ganw 0 \
                    --hard \
                    --lr 6e-5 \
                    --em 1 \
                    --update -1 \
                    --chamfer \
                    --unmask 2 \
                    --lambda_data 4 \
                    --lambda_flat 0.01 \
                    --lambda_depth 0.2 \
                    --drop 0.4 \
                    --em_gap 2 \
                    --beta1 0.9 \
                    --dataroot ./ATR/humanparsing/JPEGImages # or path/to/your/humanparsing/JPEGImages
```

- Training on Market

```sh
python train_market.py --name MKT_wgan_b48_lr0.5_em1_update-1_chf_lpl_reg0.1_data2_m2_flat7_depth0.1_drop0.4_gap2_beta0.9 \
                       --imageSize 64 \
                       --batch 48 \
                       --gan_type wgan \
                       --bg \
                       --L1 \
                       --ganw 0 \
                       --hard \
                       --lr 5e-5 \
                       --em 1 \
                       --update -1 \
                       --chamfer \
                       --lambda_data 2 \
                       --unmask 2 \
                       --lambda_flat 0.01 \
                       --lambda_depth 0.1 \
                       --drop 0.4 \
                       --amsgrad \
                       --em_gap 2 \
                       --beta1 0.9 \
                       --dataroot ./Market/hq/seg # or path/to/your/Market/hq/seg
```


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

## License
Copyright (C) 2022 NExT++ Lab. All rights reserved. Licensed under the MIT. The code is released for academic research use only. For commercial use, please contact [zdzheng@nus.edu.sg](zdzheng@nus.edu.sg).

