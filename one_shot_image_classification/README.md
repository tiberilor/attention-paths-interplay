## Code/one_shot_image_classifcation

The scripts in this folder use the experimental data to produce the figures. 
First, the experiments should be run to produce the data. See inside the ./data directory to run the experiments.
Then, run each of the .sh files in this folder to produce the associated figure.
Note that the scripts require a "dataset_location" where the dataset is contained. The dataset is a file containing both the training/test data for the task, but also the weights of the model trained with gradient descent. It can be produced using the scripts in the folder ./gradient_descent_training
