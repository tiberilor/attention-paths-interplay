#!/bin/bash

file_name="./data/all_paths/without_samples/02052024-163826_RESULTS_conv_sum_heads_bin_reg_MARKOV_OPTION_D_theory_P100_N0_200_T_30_N10_10_ParNoise1.0_PerpNoise1.0_OopNoise1.0_temp0.01_var1.0_1.0_1.0_1.0_ps0.7_0.7_0.3_0.3.pkl"

script_string="SAMPLE_posterior.py \\
#
--file_name $file_name \\
--seed 1 \\
--number_warmup 100 \\
--number_samples 100 \\
--number_skip_samples 1 \\
--number_chains 1 \\
#
--minimum_sampling_temperature 1.0e-03 \\
--sampling_device gpu \\
#
#--dataset_location <location> \\
#--overwrite_stored_samples \\"

# Use awk to filter out lines starting with #
#filtered_script=$(echo "$script_string" | awk '!/^ *#/')
filtered_script=$(echo "$script_string" | awk '!/^ *#/' | sed -e ':a' -e 'N' -e '$!ba' -e 's/\\$//')

source /.../anaconda3/bin/activate montecarlo_test

#echo "$filtered_script"

# Run the Python script using eval
eval "python $filtered_script"
