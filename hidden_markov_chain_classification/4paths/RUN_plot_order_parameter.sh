#!/bin/bash

source /.../anaconda3/bin/activate montecarlo_test

script_string="PLOT_order_parameter.py \\
# WHICH WIDTHS?
-N 10 100 1000 \\
# SAVE FIGURE
--save_figure \\
--figure_id test1 \\
# FIGURE PARAMETERS
# see at the top of the script
#
# LOCATIONS
--filenames ./data/all_paths/*.pkl \\
--dataset_location ./ \\"

# Use awk to filter out lines starting with #
filtered_script=$(echo "$script_string" | awk '!/^ *#/' | sed -e ':a' -e 'N' -e '$!ba' -e 's/\\$//')

#echo "$filtered_script"

# Run the Python script using eval
eval "python $filtered_script"
