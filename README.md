# Hyperparameter Importance Analysis for Multi-Objective AutoML
This is the implementation to the paper "Hyperparameter Importance Analysis for Multi-Objective AutoML".

*Abstract: Hyperparameter optimization plays a pivotal role in enhancing the predictive performance and generalization capabilities of ML models. However, in many applications, we do not only care about predictive performance but also about additional objectives such as inference time, memory, or energy consumption. In such multi-objective scenarios, determining the importance of hyperparameters poses a significant challenge due to the complex interplay between the conflicting objectives. In this paper, we propose the first method for assessing the importance of hyperparameters in multi-objective hyperparameter optimization. Our approach leverages surrogate-based hyperparameter importance measures, i.e., fANOVA and ablation paths, to provide insights into the impact of hyperparameters on the optimization objectives. Specifically, we compute the a-priori scalarization of the objectives and determine the importance of the hyperparameters for different objective tradeoffs. Through extensive empirical evaluations on diverse benchmark datasets with three different objective pairs, each combined with accuracy, namely time, demographic parity loss, and energy consumption, we demonstrate the effectiveness and robustness of our proposed method. Our findings not only offer valuable guidance for hyperparameter tuning in multi-objective optimization tasks but also contribute to advancing the understanding of hyperparameter importance in complex optimization scenarios.*

## Installation
```
git clone https://github.com/automl/HPI_for_MO_AutoML.git
cd HPI_for_MO_AutoML
conda create -n hpi_for_mo python=3.8
conda activate hpi_for_mo

# Install for usage
pip install .

# Install for development
make install-dev
```

Documentation at https://automl.github.io/HPI_for_MO_AutoML/main

## Run the experiments:

```
# python 
```
