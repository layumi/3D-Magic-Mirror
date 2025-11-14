Our code is tested on PyTorch 1.9.0+ and torchvision 0.10.0+. But you can try CUDA12 with latest packages. 

- Clone this repo:
```sh
git clone https://github.com/layumi/3D-Magic-Mirror.git
cd 3D-Magic-Mirror/
```


## Install with CUDA11

```sh
conda create --name magic --file spec-file.txt
conda activate magic
pip install pytorch_msssim
```

* gcc is needed by kaolin. If you have gcc about 7.3.0 - 9.5.0 (latest gcc may not work as well), please skip this step. 
(Update your gcc as follows if your gcc is too old (<=7.3) and you do not have sudo rights.)  

```sh
conda config --add channels conda-forge # add conda forge channel
#$ conda install gcc_linux-64=9.4.0 gcc_impl_linux-64=9.4.0  gxx_linux-64=9.4.0 gxx_impl_linux-64=9.4.0 # I have included in spec-file.txt
ln x86_64-conda-linux-gnu-gcc gcc # cd your_anaconda/envs/magic/bin
ln x86_64-conda-linux-gnu-g++ g++ 
```

* install Kaolin Library
(You need to mute/comment some warning like cython to install it.)
```sh
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
cd kaolin
git checkout v0.12.0
python setup.py develop
```




## Install with CUDA12 

```sh
conda create  --name magic python=3.9
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu124.html
pip install pytorch_msssim
```


Compile Pytorch3d for the latest pytorch and Cuda12. It will take some time. 
```sh
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
```

* Others: tqdm, trimesh, imageio, etc.
  
