# To Prune or Not to Prune: A Chaos-Causality Approach to Principled Pruning of Dense Neural Networks

## Abstract
Reducing the size of a neural network by removing weights without impacting its performance is a long-standing problem that has regained attention due to the need to compress large networks for mobile device usage. In the past, pruning was typically accomplished by ranking or penalizing weights based on criteria like magnitude and removing low-ranked weights before retraining the remaining ones. Not only weights, but pruning strategies may also involve removing neurons from the network in order to learn the optimal number of parameters automatically, prevent overfitting, and achieve the desired level of network size reduction. Our research approach involves formulating pruning as an optimization problem with the objective of minimizing misclassifications by selecting specific weights. To accomplish this, we have integrated the calculation of Lyapunov exponents on weight updates, enabling us to pinpoint which weight updates are chaotic in nature. In addition, we employ Granger causality to identify the weights that are responsible for misclassification in the neural network. This enables us to gain a more comprehensive understanding of which weights should be pruned to improve the network's overall performance.

## Table of Contents
- [Introduction](#introduction)
- [Pruning Strategies](#pruning-strategies)
- [Research Methodology](#research-methodology)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [Setting Up the Environment](#setting-up-the-environment)
- [License](#license)

## Introduction
This repository contains code and resources related to the paper "To prune or not to prune: A Chaos-Causality Approach to Principled Pruning of Dense Neural Networks". In this paper, we explore a novel approach to neural network pruning using chaos-causality analysis.

## Pruning Strategies
We delve into various pruning strategies employed to reduce the size of dense neural networks while maintaining their performance. This includes traditional weight magnitude-based pruning as well as more advanced strategies involving neuron removal.

## Research Methodology
Our research methodology involves formulating pruning as an optimization problem focused on minimizing misclassifications. We introduce the concept of calculating Lyapunov exponents on weight updates to identify chaotic weight updates. Additionally, we leverage Granger causality to pinpoint weights responsible for misclassifications, leading to a more informed pruning process.

## Results
In this section, we present the results of our chaos-causality approach to neural network pruning. We demonstrate the effectiveness of our method in reducing network size while preserving performance through comprehensive experiments and analysis.

## Usage
To replicate our experiments or apply our chaos-causality pruning approach to your own neural networks, refer to the testing.ipynb for our from scratch implementation. Pytorch implementation for mnist can be found in `pipeline.py` and `pipeline2.py` scripts. Detailed usage instructions and code documentation can be found there.

## Code Pipeline
- Save Weights and Accuracies for the a particular Network as shown in the testing.ipynb file.
- Calculate windowed lyapunov exponents using the MATLAB script, LE_window.m.
- Get non-causal weights by calculating granger causality between windowed lyapunov exponents and misclassification risk using the gc_test.ipynb.
- Train sparse model as shown in the testing.ipynb file.

## Lypunov Exponent Calculation
We calculate these using the [Tisean](https://www.pks.mpg.de/tisean) package in MATLAB. Install the package, put LE_window.m file in the bin folder of the package. Set path to read weights from and path to save the windowed lyapunov exponents csv to and then run it. We have fixed the overlap to 10% and windows size to 50, you can experiment with these values as well.


## Setting Up the Environment
To ensure a consistent and reproducible environment for running the code in this repository, we recommend setting up the environment using the provided `environment.yml` file. Simply run the following command:


```bash
conda env create -f environment.yml
conda activate your_environment_name
Replace your_environment_name with your desired environment name
```

