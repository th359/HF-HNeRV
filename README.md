# HF-HNeRV
### [Paper] | [DAVIS Data](https://davischallenge.org/)

## Method overview
<img src='./assets/HF-HNeRV_pipeline.png' height='450'>

## Get started
We run with Python 3.8 and CUDA 11.5, you can setup a conda environment:
```
conda create -n hfhnerv python=3.8
conda activate hfhnerv
pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 torchaudio==0.11.0 --extra-index-url  https://download.pytorch.org/whl/cu115
pip install -r requirements.txt
```

## Data Download
First, Create a data directory.
Download [DAVIS dataset](https://davischallenge.org/) and put under data directory.
```
HF-HNeRV/data/DAVIS-data
```

## High-Level structure
The code is organized as follows:
* [run.py](./run.py) includes training and validation of HF-NeRV and HNeRV in DAVIS dataset
* [train_nerv_all.py](./train_nerv_all.py) includes a training routine
* [model_all.py](./model_all.py) contains the dataloader and neural network architecture
* [data/](./data) directory video dataset, you need to put DAVIS dataset to run run.py

## Training and Evaluation
```
python run.py
```
The defalt parameter settings are as follows
```
epoch = 300
fix_epoch = 150
model_size = 1.5
filter_rate = 0.8
scale = 0.1
```

## Result
### Visualization of reconstruced vidos
<img src='./assets/cat3.png'>

## Citation

## Contact
If you have any questions, please feel free to email the authors: hayatai17@fuji.waseda.jp
