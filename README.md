## PointExplainer: Towards Transparent Parkinson's Disease Diagnosis

<img src="https://github.com/chaoxuewang/PointExplainer/blob/main/images/fig1.jpg" alt="Image text" width="300">

[arXiv](https://arxiv.org/abs/2505.03833) &nbsp; [demo](https://github.com/xc-lab/PD-Demo)


## Introduction
This work is going to appear in Information Fusion. We introduce *PointExplainer*, an interpretable diagnostic framework designed to enhance clinical interpretability and support the early diagnosis of Parkinson’s disease.

*PointExplainer* assigns attribution scores to local segments of a handwriting trajectory, highlighting their relative contribution to the model’s decision. This explanation format, consistent with expert reasoning patterns, enables clinicians to quickly identify key regions and understand the model’s diagnostic logic. In addition, we design consistency metrics to quantitatively assess the faithfulness of the explanations, reducing reliance on subjective evaluation.

In this repository, we release code and data for our *PointExplainer* diagnosis and explanation networks as well as a few utility scripts for training, testing and data processing and visualization on the SST and DST datasets.

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

## Installation
Install the required dependencies. The project requires `python=3.8` and has been tested with `pytorch=2.2.1`, `torchvision=0.17.1`, and `PyQt5=5.15.10`.  Please follow the official instructions [here](https://pytorch.org/get-started/locally/) to install PyTorch and TorchVision. Installing them with CUDA support is strongly recommended.
```
pip install -r requirements.txt
```


## Usage
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


Run the following scripts in order to complete the preprocessing pipeline:
```
# Step I: Stratified cross-validation split at the subject level
python preprocess/kfold_split.py

# Step II: Data processing (point cloud construction, etc.)
python preprocess/data_preprocess.py

# Step III: Sliding-window segmentation of handwriting trajectories
python preprocess/segment_patches.py

# Step IV: Split the training and validation sets
python preprocess/split_train_val.py
```

To train the classification model, run:
```
python train.py
```
All log files and model checkpoints will be saved automatically to the `log_dir` directory by default. You can use TensorBoard to visualize the model architecture and monitor training progress:
```
tensorboard --logdir=log_dir
```

After training, you can evaluate the model and generate visualizations of key performance metrics by running:
```
python test.py
```


A dedicated interpreter was trained for each subject, and perturbation analysis was performed to verify the reliability of the interpretation results.
```
python explanation.py
```


## License
Our code is released under MIT License (see LICENSE file for details).
