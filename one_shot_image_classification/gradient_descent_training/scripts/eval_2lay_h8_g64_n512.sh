#!/bin/bash -l
#SBATCH --account=
#SBATCH --partition=
#SBATCH --time=4:00:00
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH -o logs/%j.log

conda activate myenv

export CUDA_VISIBLE_DEVICES=0

SIZE=512
CKP=''

python ./dump_icl_pretrained.py \
  --dataset omniglot \
  --max_epoch 1 \
  --load_model_from ${CKP} \
  --work_dir './save_pretrained' \
  --min_num_training_points 2000 \
  --min_num_test_points 1000 \
  --readout_type 'mean' \
  --model_name my_dual_head_no_learning_input_constrained_linear_vit \
  --use_random_first_projection \
  --num_heads 1 \
  --num_sum_heads 8 \
  --d_model ${SIZE} \
  --dim_head ${SIZE} \
  --lr 3e-4 \
  --min_lr 1e-6 \
  --no_cls_token \
  --use_sin_pos_enc \
  --qk_dim_head 64 \
  --patch_size 8 \
  --remove_nonlinear_input_projection \
  --remove_diag_scale \
  --report_every 100 \
  --validate_every 1000 \
  --learn_attention \
  --criterion 'binary_mse' \
  --add_learned_input_layer \
  --no_residual \
  --additive_icl_label_embedding \
  --freeze_icl_label_embedding \
  --no_lr_scheduler \
  --num_layers 2
