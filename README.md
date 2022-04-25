# CNN-TDA Net
This repository contains a Python package and all experimental scripts (or notebooks) file for a CNN-TDA Net. CNN-TDA Net is a family of neural networks that take an image and the corresponding topological features as inputs and make a prediction. Here, the topological features are vectorizations for the persistence diagram of the input image. 

# Getting started
## Installation
To install the package, 

~~~
git clone https://github.com/postech-minds/cnntdanet.git
cd cnntdanet
~~~

## Training
The following command gives you a CNN-TDA Net trained on the dataset specified by the `--dataset` argument. You can find all arguments in the `train.py`.

~~~
python train.py --dataset fashion-mnist --method betti-curve --n_bins 100  
~~~

## Inference: Real-vs-bogus classification
The following command produces `result.csv` containing predictions on a set of test samples by using a trained CNN-TDA Net.

~~~
python rb_inference.py --dir_data ./datasets/test --dir_model ./checkpoints/cnn_bc_net_rb_clf.h5 --method betti-curve --n_jobs -1 --n_bins 100
~~~
