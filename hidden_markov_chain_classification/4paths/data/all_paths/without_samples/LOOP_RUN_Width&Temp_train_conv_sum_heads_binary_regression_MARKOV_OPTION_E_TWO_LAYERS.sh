#!/bin/bash

seed=7
perturbation=1.0
beta=10.0
perpendicular_noise_strength=1.0
parallel_noise_strength=1.0
# Note: the out_of_plane_noise_strength is what we call "perpendicular noise" in the paper.
# The perpendicular and parallel noise mentioned above are two directions of what we call "parallel noise" in the paper
out_of_plane_noise_strength=1.0

source /.../anaconda3/bin/activate montecarlo_test
for T in 0.01;
do
for mult in 1;
do
temp=$(echo "$T * $mult" | bc)
echo "temperature:"
echo "$temp"
for width in 10 30 50 100 300 500 1000;
do
echo "width:"
echo "$width"
script_string="TRAIN_conv_sum_heads_binary_regression_MARKOV_OPTION_E_TWO_LAYERS.py \\
#
# COMPUTATION PARAMETERS
--force_cpu \\
#
# TASK PARAMETERS
--p_a_plus 0.7 \\
--p_a_minus 0.7 \\
--p_b_plus 0.3 \\
--p_b_minus 0.3 \\
--perpendicular_noise_strength $perpendicular_noise_strength \\
--parallel_noise_strength $parallel_noise_strength \\
--out_of_plane_noise_strength  $out_of_plane_noise_strength \\
-P 100 \\
--partial_width 200 \\
-T 30 \\
#
# MODEL PARAMETERS
-N $width $width \\
--temperature $temp \\
--variances 1.0 1.0 1.0 1.0 \\
--seed 3 \\
#
# TRAIN PARAMETERS
--dont_store_checkpoint \\
-lr 0.1 \\
#--epochs 1 \\
--epochs 50000 \\
--gradient_tolerance 1e-5 \\
--scheduler_patience 100000 \\
--lr_reduce_factor 0.1 \\
--order_parameter_perturbation_strength 0.0 \\
--order_parameter_scale 1.0 \\
--order_parameter_seed 10 \\
#--hessian_test \\
#
#######################
## HEADS FIRST LAYER ##
#######################
# MAIN HEAD:
--main_head_first_layer_style same_token \\
--main_head_first_layer_shift -1.0 \\
--main_head_first_layer_beta $beta \\
--main_head_first_layer_seed $((120+seed)) \\
#--main_head_first_layer_features_perturbation 0.0 \\
#--main_head_first_layer_positions_perturbation 0.0 \\
#--main_head_first_layer_features_positions_cross_perturbation 0.0 \\
#
# UNIFORM ATTENTION:
#--betas_uniform_attention_first_layer 1.0 \\
#--seeds_uniform_attention_first_layer 110 \\
#--features_perturbation_uniform_attention_first_layer 0.0 \\
#--positions_perturbation_uniform_attention_first_layer 0.0 \\
#--features_positions_cross_perturbation_uniform_attention_first_layer 0.0 \\
#
# SAME TOKEN HEADS:
#--shifts_same_token_heads_first_layer -1.0 \\
#--betas_same_token_heads_first_layer $beta \\
#--seeds_same_token_heads_first_layer $((120+seed)) \\
#--features_perturbation_same_token_heads_first_layer $perturbation \\
#--positions_perturbation_same_token_heads_first_layer $perturbation \\
#--features_positions_cross_perturbation_same_token_heads_first_layer $perturbation \\
#
# DIFFERENT TOKEN HEADS:
#--shifts_different_token_heads_first_layer -1.0 \\
#--betas_different_token_heads_first_layer $beta \\
#--seeds_different_token_heads_first_layer $((130+seed)) \\
#--features_perturbation_different_token_heads_first_layer 0.0 \\
#--positions_perturbation_different_token_heads_first_layer 0.0 \\
#--features_positions_cross_perturbation_different_token_heads_first_layer 0.0 \\
#
# BLANK ATTENTION HEADS:
--betas_blank_attention_first_layer $beta \\
--seeds_blank_attention_first_layer $((140+seed)) \\
--features_perturbation_blank_attention_first_layer $perturbation \\
--positions_perturbation_blank_attention_first_layer $perturbation \\
--features_positions_cross_perturbation_blank_attention_first_layer $perturbation \\
#
########################
## HEADS SECOND LAYER ##
########################
# MAIN HEAD:
--main_head_second_layer_style uniform_attention \\
--main_head_second_layer_shift -1.0 \\
--main_head_second_layer_beta $beta \\
--main_head_second_layer_seed $((210+seed)) \\
#--main_head_second_layer_features_perturbation 0.0 \\
#--main_head_second_layer_positions_perturbation 0.0 \\
#--main_head_second_layer_features_positions_cross_perturbation 0.0 \\
#
# UNIFORM ATTENTION:
#--betas_uniform_attention_second_layer 1.0 \\
#--seeds_uniform_attention_second_layer $((210+seed)) \\
#--features_perturbation_uniform_attention_second_layer $perturbation \\
#--positions_perturbation_uniform_attention_second_layer $perturbation \\
#--features_positions_cross_perturbation_uniform_attention_second_layer $perturbation \\
#
# SAME TOKEN HEADS:
#--shifts_same_token_heads_second_layer \\
#--betas_same_token_heads_second_layer \\
#--seeds_same_token_heads_second_layer 220 \\
#--features_perturbation_same_token_heads_second_layer \\
#--positions_perturbation_same_token_heads_second_layer \\
#--features_positions_cross_perturbation_same_token_heads_second_layer \\
#
# DIFFERENT TOKEN HEADS:
#--shifts_different_token_heads_second_layer \\
#--betas_different_token_heads_second_layer \\
#--seeds_different_token_heads_second_layer 230 \\
#--features_perturbation_different_token_heads_second_layer \\
#--positions_perturbation_different_token_heads_second_layer \\
#--features_positions_cross_perturbation_different_token_heads_second_layer \\
#
# BLANK ATTENTION HEADS:
--betas_blank_attention_second_layer $beta \\
--seeds_blank_attention_second_layer $((240+seed)) \\
--features_perturbation_blank_attention_second_layer $perturbation \\
--positions_perturbation_blank_attention_second_layer $perturbation \\
--features_positions_cross_perturbation_blank_attention_second_layer $perturbation \\
#
# STORAGE / PRINT PARAMETERS
--dont_store_checkpoint \\
# --dont_store_scalars
# --dont_store_tensors
--number_steps_store_scalars 500 \\
--number_steps_store_tensors 500 \\
--number_steps_store_checkpoint 500 \\
--number_steps_print_info 50 \\
--results_storage_location ./ \\
#--results_id TEST_ \\
#
# PLOTS / CHECKS (exits before training)
#--plot_pre_kernels \\
#--plot_pre_kernels_style diagonal \\
#--plot_kernel \\
#--plot_pre_kernels_style all \\
#--plot_pre_kernels_with_temperature \\
#--plot_kernel \\
#--plot_kernel_with_temperature \\
#--test_kernel_invertibility \\
#--plot_kernel_invertibility \\
#--test_kernel_invertibility_with_temperature \\
#
# values to remember
#--epochs 50000 \\"

# Use awk to filter out lines starting with #
#filtered_script=$(echo "$script_string" | awk '!/^ *#/')
filtered_script=$(echo "$script_string" | awk '!/^ *#/' | sed -e ':a' -e 'N' -e '$!ba' -e 's/\\$//')

source /.../anaconda3/bin/activate montecarlo_test

#echo "$filtered_script"

# Run the Python script using eval
eval "python $filtered_script"

done
done
done
