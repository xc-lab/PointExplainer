## PointExplainer: Towards Transparent Parkinson's Disease Diagnosis

<img src="https://github.com/chaoxuewang/PointExplainer/blob/main/images/fig1.jpg" alt="Image text" width="300">

[arXiv](https://arxiv.org/abs/2505.03833) &nbsp;&nbsp;&nbsp;&nbsp; [arXiv](https://arxiv.org/abs/2505.03833)

## Introduction
We proposed an explainable diagnosis framework combined with a digitized hand-drawn test for Parkinson's disease diagnosis.

Deep neural networks have shown potential in analyzing digitized hand-drawn signals for diagnosing Parkinson's disease. However, the lack of interpretability in most existing methods poses a challenge to building user trust. We propose an explainable diagnosis framework, named *PointExplainer*, for providing personalized predictions suitable for human reasoning. *PointExplainer* assigns importance values to hand-drawn segments, reflecting their relative contribution to the model's decision. We also verify model behavior consistency to address the issue of faithfulness. 

In this repository, we release code for training *PointExplainer* and provide a simple demo showing the personalized inference.

## Installation
1. Clone this repository.
2. Install dependencies. The code requires `python=3.8`, as well as `pytorch=2.2.1`, `torchvision=0.17.1` and `PyQt5=5.15.10`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.
   ```
   pip install -r requirements.txt
   ```
3. Go to the next step **Getting Started**.



## Getting Started
### 1. Dataset
Download the ParkinsonHW dataset [here](https://archive.ics.uci.edu/dataset/395/parkinson+disease+spiral+drawings+using+digitized+graphics+tablet), which includes two different methods (SST and DST) for testing hand-drawn Archimedean spiral patterns. Note that the download path should be `./diagnosis/data/raw_data`, and the healthy subject data is placed in the `KT` subfolder, and the Parkinson's patient data is placed in the `PD` subfolder.

<img src="https://github.com/chaoxuewang/PointExplainer/blob/main/images/fig4.jpg" alt="Image text" width="300">


### 2. Diagnosis
#### 2.1 preprocessing
Run the following codes in sequence to complete data preprocessing:
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
Train a model to classify hand-drawn data:
```
python diagnosis/train.py
```
Log files and network parameters will be saved to `diahnosis/log_dir` folder in default. We can use TensorBoard to view the network architecture and monitor the training progress.
```
tensorboard --logdir=diagnosis/log_dir
```

#### 2.3 testing
After the above training, we can test the model and output some visualizations of the metric curves. You can run the evaluation code with:
```
python diagnosis/test.py
```


### 3. Explanation
We have uploaded the trained weight lists (i.e., explanations).

## Demo
We provide a simple demo that illustrates how you can use \textit{PointExplainer} to make explainable personalized predictions.


## Citation
If you find our work useful in your research, please consider citing:

## Contributing
Contributions to this repository are welcome. Examples of things you can contribute:
- Analyze the weight list results and try to find the commonalities between hand-drawn points with the same attributes.
- Speed Improvements. Like re-writing some Python code in TensorFlow or Cython.



## License
Our code is released under MIT License (see LICENSE file for details).
