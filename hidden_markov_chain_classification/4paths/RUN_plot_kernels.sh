#!/bin/bash

source /home/lorenzo/anaconda3/bin/activate montecarlo_test

script_string="PLOT_kernels.py \\
# SAVE FIGURE
--save_figure \\
--figure_id test1 \\
# FIGURE PARAMETERS
# see at the top of the script
#
# LOCATIONS
--filename ./data/all_paths/without_samples/*N10_*.pkl \\
--dataset_location ./ \\"

# Use awk to filter out lines starting with #
filtered_script=$(echo "$script_string" | awk '!/^ *#/' | sed -e ':a' -e 'N' -e '$!ba' -e 's/\\$//')

#echo "$filtered_script"

# Run the Python script using eval
eval "python $filtered_script"
