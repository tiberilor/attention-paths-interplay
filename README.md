## Code/Supplemental material 

This is the code used to produce experimental results for the arXiv preprint "Dissecting the Interplay of Attention Paths in a Statistical Mechanics Theory of Transformers" 

The code is organized into two main folders, one for each experiment presented in the paper, i.e. hidden markov chain 
classification, and one-shot image classification. Inside each folder, we give the code and instructions to reproduce 
all of the figures presented in the main text.

### Contents
* `hidden_markov_chain_classification` contains code for reproducing the hidden markov chain classification experiment.
* `one_shot_image_classification` contains code for reproducing the one-shot image classification experiment.

### Requirement

- PyTorch nightly is needed because of precision-64 Adam implementation (install instruction/commandline from https://pytorch.org/get-started/locally/)
- Unless specified otherwhise, see `montecarlo_test.yml` for required packages
- We used Python 3.10 for all our experiments

