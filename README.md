# PointExplainer
PointExplainer: Towards Practical Transparent Parkinson's Disease Diagnosis


<img src="https://github.com/chaoxuewang/PointExplainer/blob/main/images/fig1.jpg" alt="Image text" width="300">

Created by Xuechao Wang at Ghent University.

## Introduction
This work is based on our arXive report, which is going to appear in [Medical Image Anglysis](https://www.sciencedirect.com/journal/medical-image-analysis). We proposed an explainable diagnosis framework combined with a digitized hand-drawn test for Parkinson's disease diagnosis.

Deep neural networks have shown potential in analyzing digitized hand-drawn signals for diagnosing Parkinson's disease. However, the lack of interpretability in most existing methods poses a challenge to building user trust. We propose an explainable diagnosis framework, named *PointExplainer*, for providing personalized predictions suitable for human reasoning. *PointExplainer* assigns importance values to hand-drawn segments, reflecting their relative contribution to the model's decision. We also verify model behavior consistency to address the issue of faithfulness. 

In this repository, we release code for testing *PointExplainer* and provide a simple demo showing the personalized inference.

## Installation
The code requires `python=3.8`, as well as `pytorch=2.2.1` and `torchvision=0.17.1`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.


## Getting Started
### 1. Dataset
Download the ParkinsonHW dataset [here](https://archive.ics.uci.edu/dataset/395/parkinson+disease+spiral+drawings+using+digitized+graphics+tablet), which includes two different methods for testing hand-drawn Archimedean spiral patterns. Note that the download path should be `./diagnosis/data/raw_data`, and the healthy subject data is placed in the `KT` subfolder, and the Parkinson's patient data is placed in the `PD` subfolder.

### 2. Diagnosis
#### 2.1 preprocessing
Run the following commands in sequence to complete data preprocessing:
```
#step I: Stratified cross-validation segmentation at the individual level
python diagnosis/preprocess/kfold_split.py
#step II: Data processing (constructing point clouds, data enhancement...)
python diagnosis/preprocess/data_augmentation.py
#step III: Sliding window segmentation
python diagnosis/preprocess/segment_patches.py
#step IV: Divide the training set and the validation set
python diagnosis/preprocess/split_train_val.py
```

#### 2.2 training
We are providing pre-trained weights for SST and DST datasets to make it easier to start.

You can also run the evaluation code with:

#### 2.3 testing

### Explanation

## Demo
We provide a simple demo with PyQt5 that illustrates how you can use \textit{PointExplainer} for explaable personalized predictions.


## Citation
If you find our work useful in your research, please consider citing:

## Contributing
Contributions to this repository are welcome. Examples of things you can contribute:


## License
Our code is released under MIT License (see LICENSE file for details).
