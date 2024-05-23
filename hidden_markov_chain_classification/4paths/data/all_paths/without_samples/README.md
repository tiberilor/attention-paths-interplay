## Code/4paths

To produce the data:
1. Run "LOOP_RUN_Width&Temp_train_conv_sum_heads_binary_regression_MARKOV_OPTION_E_TWO_LAYERS.sh". This will produce data files containing the model at varying N, and the result of the numerical evaluation of the order parameter.
2. Run "RUN_sample_posterior.sh" on each of the prodcued data files. This will produce a new copy of the data file, which contains in addition the sampled weights. This script may need to be run on a cluster, possibly on GPU.
3. Move the obtained data files to "../"

