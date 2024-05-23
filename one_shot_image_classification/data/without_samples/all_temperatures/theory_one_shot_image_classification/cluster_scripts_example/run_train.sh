#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

mkdir -p logs results figures run_logs final_results

CODE=""
DATA=""
DATA_DIR=""

P=$1
N=$2
T=$3

RESULTS_FILE="./results/P${P}_N${N}_T${T}__${DATA}"
FIGURE="./figures/P${P}_N${N}_T${T}__${DATA}.png"
TRAINLOG="./run_logs/P${P}_N${N}_T${T}__${DATA}.train.log"
VIEWLOG="./run_logs/P${P}_N${N}_T${T}__${DATA}.view.log"

FINAL_RESULTS="./final_results/P${P}_N${N}_T${T}.txt"

echo "[bash] $P, $N, $T"
### Training

python ${CODE}/TRAIN_conv_sum_heads_images_binary_regression_PRETRAINED_HEADS.py \
 --dataset ${DATA} \
 --output_log_file_name ${TRAINLOG} \
 --token_readout_style average_pooling \
 -P ${P} \
 -N ${N} \
 --temperature ${T} \
 --variances 1.0 1.0 1.0 1.0 \
 -lr 0.008 \
 --epochs 50000 \
 --gradient_tolerance "1.0e-05" \
 --scheduler_patience 100000 \
 --lr_reduce_factor 0.1 \
 --minimum_learning_rate 0.001 \
 --order_parameter_perturbation_strength 0.0 \
 --order_parameter_scale 1.0 \
 --order_parameter_seed 10 \
 --number_steps_store_scalars 1000000 \
 --number_steps_store_tensors 1000000 \
 --number_steps_store_checkpoint 1000000 \
 --number_steps_print_info 1 \
 --dont_store_checkpoint \
 --results_file_name ${RESULTS_FILE} \
 --results_storage_location "./results/" \
 --dataset_location ${DATA_DIR} \
 --force_cpu

### Testing
echo "[bash] --- end of training ---"
echo "[bash] --- start testing ---"

python ${CODE}/VIEW_training_results.py \
 -f ${RESULTS_FILE} \
 --output_log_file_name ${VIEWLOG} \
 --disable_figure_visualization \
 --dataset_location ${DATA_DIR} \
 --test_predictor \
 --number_test_examples 1000 \
 --plot_order_parameter \
 --plot_order_parameter_file_name ${FIGURE}

echo "[bash] --- end testing ---"
echo "[bash] --- gather results ---"

echo $P, $N, $T > ${FINAL_RESULTS}
echo "Renorm:" >> ${FINAL_RESULTS}
cat ${VIEWLOG} | grep "Renorm, classification accuracy (mean predictor only)" >> ${FINAL_RESULTS}
echo "GP:" >> ${FINAL_RESULTS}
cat ${VIEWLOG} | grep "GP, classification accuracy (mean predictor only)" >> ${FINAL_RESULTS}
echo "[bash] --- end ---"
