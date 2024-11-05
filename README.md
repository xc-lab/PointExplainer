# PointExplainer
PointExplainer: Towards Practical Transparent Parkinson's Disease Diagnosis

<img src="https://github.com/chaoxuewang/PointExplainer/blob/main/images/fig1.jpg" alt="Image text" width="300">

## Introduction
This work is based on our arXive report, which is going to appear in Medical Image Anglysis. We proposed an explainable diagnosis framework for diagnosing Parkinson's disease.

Deep neural networks have shown potential in analyzing digitized hand-drawn signals for diagnosing Parkinson's disease. However, the lack of interpretability in most existing methods poses a challenge to building user trust. In this paper, we propose an explainable diagnosis framework, named \textit{PointExplainer}, for providing personalized predictions. \textit{PointExplainer} assigns importance values to hand-drawn segments, reflecting their relative contribution to the model's decision. Its novel components include: (1) encoding hand-drawn signals into 3D point clouds to represent hand-drawn trajectories, (2) training an interpretable surrogate model to mimic the behavior of a black-box diagnosis model. We also verify model behavior consistency to address the issue of faithfulness. 

In this repository, we release code (on public dataset-ParkinsonHW) for traning a PD diagnosis network, as well as for providing intelligible explanations.
