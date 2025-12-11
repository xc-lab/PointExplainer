## PointExplainer: Towards Transparent Parkinson's Disease Diagnosis

<img src="https://github.com/chaoxuewang/PointExplainer/blob/main/images/fig1.jpg" alt="Image text" width="300">

[arXiv](https://arxiv.org/abs/2505.03833) &nbsp; [demo](https://github.com/xc-lab/PD-Demo)

## Citation
If you find our work useful in your research, please consider citing:
```bibtex
@article{wang2025pointexplainer,
  title={PointExplainer: Towards Transparent Parkinson's Disease Diagnosis},
  author={Wang, Xuechao and Nomm, Sven and Huang, Junqing and Medijainen, Kadri and Toomela, Aaro and Ruzhansky, Michael},
  journal={arXiv preprint arXiv:2505.03833},
  year={2025}
}
```

## Introduction
We introduce *PointExplainer*, an interpretable diagnostic framework designed to enhance clinical interpretability and support the early diagnosis of Parkinson’s disease.

*PointExplainer* assigns attribution scores to local segments of a handwriting trajectory, highlighting their relative contribution to the model’s decision. This explanation format, consistent with expert reasoning patterns, enables clinicians to quickly identify key regions and understand the model’s diagnostic logic. In addition, we design consistency metrics to quantitatively assess the faithfulness of the explanations, reducing reliance on subjective evaluation.

In this repository, we release code and data for our *PointExplainer* diagnosis and explanation networks as well as a few utility scripts for training, testing and data processing and visualization on the SST and DST datasets.

## Installation
Install the required dependencies. The project requires `python=3.8` and has been tested with `pytorch=2.2.1`, `torchvision=0.17.1`, and `PyQt5=5.15.10`.  Please follow the official instructions [here](https://pytorch.org/get-started/locally/) to install PyTorch and TorchVision. Installing them with CUDA support is strongly recommended.
```
bash
pip install -r requirements.txt
```


## Getting Started
### 1. Dataset
Download the dataset from [here](https://archive.ics.uci.edu/dataset/395/parkinson+disease+spiral+drawings+using+digitized+graphics+tablet).
The dataset contains two handwriting patterns, SST (Static Spiral Test) and DST (Dynamic Spiral Test), used for acquiring digitized Archimedean spiral drawings.
After downloading, organize the dataset into the following directory structure:
```
data/
└── ParkinsonHW/
    └── raw_data/
        ├── KT/   # healthy control subjects
        └── PD/   # Parkinson’s disease patients
```

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
