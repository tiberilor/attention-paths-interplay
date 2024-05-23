#!/bin/bash

source /.../anaconda3/bin/activate montecarlo_test

script_string="PLOT_classification_accuracy_mean_predictor.py \\
# SAVE FIGURE
--save_figure \\
#--figure_id ID \\
# FIGURE PARAMETERS
# see at the top of the script
#
# LOCATIONS
--filenames ./data/all_paths/*.pkl \\
--good_path_filename ./data/good_path/*.pkl \\
--good_and_denoising_paths_filename ./data/good_and_denoising_paths/*.pkl \\
--dataset_location ./ \\
#
# TEST PARAMETERS
--number_test_examples 1000 \\
#
# COMPUTATION PARAMETERS
--examples_chunk_size 10 \\"

# Use awk to filter out lines starting with #
filtered_script=$(echo "$script_string" | awk '!/^ *#/' | sed -e ':a' -e 'N' -e '$!ba' -e 's/\\$//')

#echo "$filtered_script"

# Run the Python script using eval
eval "python $filtered_script"