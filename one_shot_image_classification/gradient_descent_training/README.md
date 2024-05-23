# Code for Gradient Descent Training

### Requirement

- See `environment.sgd.yml` for required packages
- We used Python 3.10 for all our experiments


### General instructions

- See script files under `scripts` to train/eval/prune models.
- `eval` scripts are used to dump a pre-trained model file with a format which is expected by the theory code.
- the main code file for the experiments for one-shot image classification is `main_icl.py`

### Acknowledgements

Our codebase includes code from the following repositories:

* This codebase was originally forked and modified from the ViT implementation of [omihub777/ViT-CIFAR](https://github.com/omihub777/ViT-CIFAR). Note that, consequently, our code contains extra options that are not used in our experiments presented in the paper (e.g., data augmentation).

* Code under `torchmeta_local` are forked and modified from [tristandeleu/pytorch-meta](https://github.com/tristandeleu/pytorch-meta) as a standard few-shot learning data preparation/processing and dataloader pipeline.

* Parts of code in `main_icl.py` are taken from [IDSIA/modern-srwm](https://github.com/IDSIA/modern-srwm/tree/main/supervised_learning)

* Parts of code in `my_vit.py` are taken from [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py)

* Parts of code (specified in the code files) are taken from [IDSIA/recurrent-fwp](https://github.com/IDSIA/recurrent-fwp/blob/master/algorithmic/listops_data.py)
