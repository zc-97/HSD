# Hierarchical Item Inconsistency Signal learning for Sequence Denoising in Sequential Recommendation
***
PyTorch implementation of the paper "Hierarchical Item Inconsistency Signal learning for Sequence
Denoising in Sequential Recommendation", CIKM.2022
***
## Abstract
Sequential recommender systems aim to recommend the next item that the target users are most interested in, based on users' historical interaction sequences.
However, historical sequences typically contain some inherent noise (e.g., accidental interactions), which is noisy for learning sequence representations and thus mislead the next-item recommendation. 
In addition, the absence of supervised signals, i.e., labels indicating noisy items, makes the problem of sequence denoising rather challenging.
To this end, we propose a novel sequence denoising paradigm in sequential recommendations by learning hierarchical item inconsistency signals. 
To be specific, we design a hierarchical sequence denoising (HSD) model, which first learns the two levels of inconsistency signals within input sequences, and then generates noiseless subsequences (i.e., dropping inherent noisy items) for subsequent sequential recommenders. 
It is noteworthy that HSD is flexible to accommodate supervised item signals and can be seamlessly integrated into most existing sequential recommenders to boost their effectiveness. 
Extensive experiments on five public benchmark datasets demonstrate the superiority of HSD over state-of-the-art denoising methods, and its applicability over a wide variety of mainstream sequential recommendation models. 
**We will publicize the implementation code to foster the research on this important topic.**

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