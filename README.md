# Hierarchical Item Inconsistency Signal learning for Sequence Denoising in Sequential Recommendation
***
PyTorch implementation of the paper "Hierarchical Item Inconsistency Signal learning for Sequence
Denoising in Sequential Recommendation", CIKM.2022
***

## Requirements
***
Our model HSD is implemented based on the RecBole v1.0.1. Both the processing of the dataset and the metrics calculation follow the implementation of RecBole.
* python 3.70+
* PyTorch 1.7.1+
* yaml 6.0+
* openpyxl 3.0.9+
* RecBole 1.0.1+
* tqdm 4.64.0

## Preparing Environment
***
### Install [Recbole](https://github.com/RUCAIBox/RecBole), 
#### Install from Conda
```commandline
conda install -c aibox recbole
```
#### Install from pip
```commandline
pip install recbole
```
#### Install from source
```commandline
git clone https://github.com/RUCAIBox/RecBole.git && cd RecBole
pip install -e . --verbose
```
### Usage

## Citation
If you use this code, please cite the paper.
