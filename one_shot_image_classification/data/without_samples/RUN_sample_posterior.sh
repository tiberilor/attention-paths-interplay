#!/bin/bash

file_name="P600_N500_T0.25__apr14_incontext_v0_layer2_channel3_patch8_h4_g128_pooling_pretrained.pkl"

script_string="SAMPLE_posterior.py \\
#
--file_name $file_name \\
--seed 4 \\
--number_warmup 20 \\
--number_samples 20 \\
--number_skip_samples 1 \\
--number_chains 10 \\
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
