## Prerequisites
```
conda create -n moa python=3.8
conda activate moa
```
If your gcc version is < 5.0, first install:
```
conda install https://anaconda.org/brown-data-science/gcc/5.4.0/download/linux-64/gcc-5.4.0-0.tar.bz2
```

Then,
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
pip install -e ./
pip install wandb
```