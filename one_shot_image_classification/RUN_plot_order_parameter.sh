#!/bin/bash

source /.../anaconda3/bin/activate montecarlo_test

script_string="PLOT_order_parameter.py \\
# SAVE FIGURE
--save_figure \\
--figure_id test1 \\
# FIGURE PARAMETERS
# see at the top of the script
#
# LOCATIONS
--filename_gibbs ./data/P600_N10_T0.5__apr14_incontext_v0_layer2_channel3_patch8_h4_g128_pooling_pretrained_PLUS_SAMPLES_05052024-171040_seeds1_Nw100_Ns100_Nc10.pkl \\
--filename_gradient_descent ./data/gradient_descent_trained_model/apr17_fuly_trained_2layer_h4_incontext_v0.pt \\
--dataset_location ./ \\"

# Use awk to filter out lines starting with #
filtered_script=$(echo "$script_string" | awk '!/^ *#/' | sed -e ':a' -e 'N' -e '$!ba' -e 's/\\$//')

#echo "$filtered_script"

# Run the Python script using eval
eval "python $filtered_script"
