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
pip install -e ./
```