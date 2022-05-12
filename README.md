[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) ![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/github/layumi/3D-Magic-Mirror.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/github/layumi/3D-Magic-Mirror/context:python)

## 3D Magic Mirror: Clothing Reconstruction from a Single Image via a Causal Perspective
![](./Figure7.gif)

![](./Figure10.gif)

<img src="./Figure8.gif" style="zoom:50%;" />

[[Project]](https://zdzheng.xyz/publication/3D-Ma2022) [[Code]](https://github.com/layumi/3D-Magic-Mirror) [[Paper]](https://arxiv.org/abs/2204.13096)

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
$ conda create --name smr --file spec-file.txt
$ conda activate smr
$ pip install tensorboard tensorboardx
$ pip install opencv-python imageio trimesh typing-extensions
$ pip install timm prettytable h5py imgaug smplx munkres joblib pycocotools lap plotly pandas
$ pip install ipywidgets keyboard transforms3d chumpy
```

* Update your gcc if your gcc is too old  and you do not have sudo rights.

```sh
$ conda config --add channels conda-forge # add conda forge channel
$ conda install gcc_linux-64=9.4.0 gcc_impl_linux-64=9.4.0
$ conda install gxx_linux-64=9.4.0 gxx_impl_linux-64=9.4.0
$ ln x86_64-conda-linux-gnu-gcc gcc # cd bin
$ ln x86_64-conda-linux-gnu-g++ g++ 
```

* Kaolin Library

```sh
$ git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
$ git checkout v0.9.1
$ python setup.py develop
```

* Others: tqdm, trimesh, imageio, etc.


``` sh
$ python -m ROMP.romp.predict.image_simple
```

Our code is tested on PyTorch 1.9.0+ and torchvision 0.10.0+.

### Dataset Preparation

Download the dataset:

- Market-HQ

Download the processed data from [Google Drive](https://drive.google.com/file/d/10gNi3n8Iny4O4MOZRs5dNFICfj8ri9eW/view?usp=sharing). Or use the gdrive tool to download:

```
gdrive download 10gNi3n8Iny4O4MOZRs5dNFICfj8ri9eW
```

- Market-Seg

Download the processed data from [Google Drive](https://drive.google.com/file/d/1TxKpVMiRjYknQyf0MJthvzSWfPJ0h6fm/view?usp=sharing). Or use the gdrive tool to download:

```
gdrive download 1TxKpVMiRjYknQyf0MJthvzSWfPJ0h6fm
```


- CUB

Download the processed data from [Google Drive](https://drive.google.com/file/d/1SkX_FWUfLOaTr371TBkQnDH9oDJ5Khwc/view?usp=sharing). Or use the gdrive tool to download:

```
gdrive download 1SkX_FWUfLOaTr371TBkQnDH9oDJ5Khwc
```

- ATR

Download the processed data from [Google Drive](https://drive.google.com/file/d/1kpsMDrbM4FQqtP7Y1nKslp4OlRKNvbaL/view). Or use the gdrive tool to download:

```
gdrive download 1kpsMDrbM4FQqtP7Y1nKslp4OlRKNvbaL
```



Preparation: put the images with the same id in one folder. You may use 
```bash
python prepare_market.py          # for Market-1501
python prepare_ATR.py             # for ATR
python prepare_cub_edge.py        # for CUB
python preproces_cub.py           # for CUB
```
Note to modify the dataset path to your own path.

### Testing

#### Download the trained model
We provide our trained model. You may download it from [Google Drive](https://drive.google.com/open?id=1lL18FZX1uZMWKzaZOuPe3IuAdfUYyJKH) (or [Baidu Disk](https://pan.baidu.com/s/1503831XfW0y4g3PHir91yw) password: rqvf). You may download and move it to the `outputs`.
```
├── outputs/
│   ├── E0.5new_reid0.5_w30000
├── models
│   ├── best/                   
```

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
                --beta1 0.9
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
                    --beta1 0.9
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
                       --beta1 0.9
```

## Citation

Please cite this paper if it helps your research:

```bibtex
@article{zheng20223d,
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

