import argparse
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from einops import rearrange

from utils import get_model
from einops import repeat

from utils_text_data import ImdbDataset

import sys
import os
import json
import logging
import hashlib
import time


parser = argparse.ArgumentParser()
parser.add_argument("--show_progress_bar", action="store_true")
parser.add_argument("--dataset", default="omniglot", type=str,
                    help="[omniglot, miniimagenet]")
parser.add_argument("--data_dir", default="./data", type=str)
parser.add_argument("--valid_set_size", default=1000, type=int)
parser.add_argument("--num_classes", default=1, type=int)
parser.add_argument("--model_name", default="my_vit", type=str)
parser.add_argument("--patch_size", default=8, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--eval_batch_size", default=1024, type=int)
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--pad_idx", default=50256, type=int)  # RoBERTa
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--min_lr", default=1e-5, type=float)
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--off-benchmark", action="store_true")
parser.add_argument("--max_epochs", default=200, type=int)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--weight_decay", default=5e-5, type=float)
parser.add_argument("--warmup_epoch", default=5, type=int)
parser.add_argument("--precision", default=16, type=int)
parser.add_argument("--gradient_clip", default=0.0, type=float)
parser.add_argument('--work_dir', default='save_models', type=str,
                    help='where to save model ckpt.')
parser.add_argument("--report_every", default=0, type=int,
                    help='report stats on train every this batch.')
parser.add_argument("--validate_every", default=0, type=int,
                    help='report stats on train every this batch.')
parser.add_argument("--binary_task", action="store_true",
                    help='define the task as binary classi 0-4 v 5-9.')
parser.add_argument("--binary_task_only_two_classes", action="store_true",
                    help='define the task as binary classi 0 v 1.')
parser.add_argument("--binary_class_choice", default="01", type=str)
# CIFAR-10 class label order:
# ['airplane', 'automobile', 'bird', 'cat', 'deer',
#  'dog', 'frog', 'horse', 'ship', 'truck']
parser.add_argument("--add_learned_input_layer", action="store_true")
parser.add_argument("--readout_type", default="cls", type=str,
                    choices=['cls', 'mean'])
parser.add_argument("--input_vocab_size", default=50265, type=int)  # RoBERTa 50265 vs 50257
parser.add_argument("--token_embedding_dim", default=1024, type=int)  # RoBERTa
parser.add_argument("--num_workers", default=None)

# dumping related
parser.add_argument('--load_model_from', default='./best_model.pt', type=str,
                    help='path to the model checkpoint.')
parser.add_argument("--min_num_training_points", default=0, type=int)
parser.add_argument("--min_num_test_points", default=0, type=int)
parser.add_argument("--min_num_extra_test_points", default=0, type=int)

# loss related
parser.add_argument("--criterion", default="ce")
parser.add_argument("--label-smoothing", action="store_true")
parser.add_argument("--smoothing", default=0.1, type=float)

# image processing
parser.add_argument("--rcpaste", action="store_true")
parser.add_argument("--cutmix", action="store_true")
parser.add_argument("--mixup", action="store_true")
parser.add_argument("--autoaugment", action="store_true")

# model params
parser.add_argument("--num_layers", '-L', default=7, type=int)
parser.add_argument("--d_model", '-N', default=384, type=int)
parser.add_argument("--num_heads", '-H', default=12, type=int)
parser.add_argument("--dim_head", '-M', default=32, type=int)
parser.add_argument("--qk_dim_head", '-G', default=32, type=int)
parser.add_argument("--num_sum_heads", '-F', default=1, type=int)
parser.add_argument("--concat_pos_emb_dim", '-K', default=96, type=int)
# used only when `--concat_pos_enc` is used; if positional encoding is
# concatenated, patch projection dim is = d_model - concat_pos_emb_dim

# model options
parser.add_argument("--use_parallel", action="store_true")
parser.add_argument("--no_residual", action="store_true")
parser.add_argument("--no_cls_token", action="store_true")
parser.add_argument("--use_random_first_projection", action="store_true")
parser.add_argument("--use_sin_pos_enc", action="store_true")
parser.add_argument("--use_random_position_encoding", action="store_true")
parser.add_argument("--concat_pos_enc", action="store_true")
parser.add_argument("--remove_diag_scale", action="store_true",
                    help='remove scaler on the diagonal.')
parser.add_argument("--learn_attention", action="store_true")
parser.add_argument("--remove_diagonal_init", action="store_true")
parser.add_argument("--remove_nonlinear_input_projection", action="store_true")
parser.add_argument("--readout_column_index", default=0, type=int)
parser.add_argument("--freeze_value", action="store_true")
parser.add_argument("--freeze_input_projection", action="store_true")
parser.add_argument("--freeze_readout", action="store_true")
parser.add_argument('--init_model_from', default=None, type=str,
                    help='e.g. save_models/aaa/best_model.pt.')
parser.add_argument("--init_also_value", action="store_true")
parser.add_argument("--init_entire_model", action="store_true")
parser.add_argument("--init_also_readout", action="store_true")
parser.add_argument("--save_init_model", action="store_true")
parser.add_argument("--add_biases", action="store_true",
                    help='Add biases to input projection and readout.')

# model params for the baseline
parser.add_argument("--d_ff", default=384, type=int)
parser.add_argument("--dropout", default=0.0, type=float)
parser.add_argument("--vit_no_feedforward", action="store_true")

# other params (deprecated; to be removed)
parser.add_argument("--equal_head_dim_model_dim", action="store_true")
parser.add_argument("--low_rank_qk", action="store_true")

# other
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--icl_eval_num_iter", default=100, type=int)
parser.add_argument("--icl_num_shots", default=1, type=int)
parser.add_argument("--additive_icl_label_embedding", action="store_true")
parser.add_argument("--freeze_icl_label_embedding", action="store_true")

parser.add_argument("--no_lr_scheduler", action="store_true")
parser.add_argument("--augment_omniglot", action="store_true")
parser.add_argument("--use_grey_scale", action="store_true")

parser.add_argument("--drop_heads", type=str, default=None,
                    help='comma separated e.g. "0-1,0-2" heads to be removed for eval; layerID-headID')

# for wandb
parser.add_argument('--project_name', type=str, default="vit11",
                    help='project name for wandb.')
parser.add_argument('--job_name', type=str, default=None,
                    help='job name for wandb.')
parser.add_argument('--use_wandb', action='store_true',
                    help='use wandb.')

args = parser.parse_args()

args.benchmark = True if not args.off_benchmark else False
args.gpus = torch.cuda.device_count()
if args.num_workers is None:
    args.num_workers = 4*args.gpus if args.gpus else 8
else:
    args.num_workers = int(args.num_workers)
args.is_cls_token = True if not args.no_cls_token else False

if not args.gpus:
    args.precision = 32

# set in_context_learning
args.in_context_learning = True
args.criterion = 'binary_mse'

exp_str = ''
for arg_key in vars(args):
    exp_str += str(getattr(args, arg_key)) + '-'

# taken from https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
exp_hash = str(int(hashlib.sha1(exp_str.encode("utf-8")).hexdigest(), 16) % (10 ** 8))

# Set work directory
job_str = (f"{args.model_name[:8]}-{args.criterion}-L{args.num_layers}-N{args.d_model}-"
           f"H{args.num_heads}-M{args.dim_head}-G{args.qk_dim_head}-"
           f"F{args.num_sum_heads}-K{args.concat_pos_emb_dim}-"
           f"randomIP{args.use_random_first_projection}-"
           f"nores{args.no_residual}-nocls{args.no_cls_token}-"
           f"sinpos{args.use_sin_pos_enc}-concatpos{args.concat_pos_enc}-"
           f"randompos{args.use_random_position_encoding}-"
           f"learnAttn{args.learn_attention}-para{args.use_parallel}-"
           f"LearnedInput{args.add_learned_input_layer}-"
           f"binary{args.binary_class_choice}")

args.work_dir = os.path.join(
    args.work_dir, f"{job_str}-{exp_hash}-{time.strftime('%Y%m%d-%H%M%S')}")
if not os.path.exists(args.work_dir):
    os.makedirs(args.work_dir)

work_dir_key = '/'.join(os.path.abspath(args.work_dir).split('/')[-3:])

# logging
log_file_name = f"{args.work_dir}/log.txt"
handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
logging.basicConfig(
    level=logging.INFO, format='%(message)s', handlers=handlers)

loginf = logging.info
loginf("== Text Classification ==")

loginf(f"Command executed: {sys.argv[:]}")
loginf(f"Args: {json.dumps(args.__dict__, indent=2)}")

loginf(f"torch version: {torch.__version__}")
loginf(f"Work dir: {args.work_dir}")
loginf(f"Seed: {args.seed}")

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

valid_seed = seed
test_seed = seed
loginf(f"Valid seed: {valid_seed}, Test seed: {test_seed}")

batch_size = args.batch_size
eval_batch_size = args.eval_batch_size
data_path = args.data_dir

train_pos_file = f"{data_path}/train.pos.bpe.txt"
train_neg_file = f"{data_path}/train.neg.bpe.txt"

valid_pos_file = f"{data_path}/valid.pos.bpe.txt"
valid_neg_file = f"{data_path}/valid.neg.bpe.txt"

# Construct dataset
train_data = ImdbDataset(
    pos_data_file=train_pos_file, neg_data_file=train_neg_file,
    max_seq_length=args.max_seq_length, pad_idx=args.pad_idx)

# valid_data = ImdbDataset(
#     pos_data_file=valid_pos_file, neg_data_file=valid_neg_file,
#     max_seq_length=args.max_seq_length, pad_idx=args.pad_idx)

# Set dataloader
train_dl = DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True)
# valid_dl = DataLoader(
#     dataset=valid_data, batch_size=eval_batch_size, shuffle=False)

# test set
test_pos_file = f"{data_path}/test.pos.bpe.txt"
test_neg_file = f"{data_path}/test.neg.bpe.txt"
test_data = ImdbDataset(
    pos_data_file=test_pos_file, neg_data_file=test_neg_file,
    max_seq_length=args.max_seq_length, pad_idx=args.pad_idx)
test_dl = DataLoader(
    dataset=test_data, batch_size=eval_batch_size, shuffle=True)

# Set work directory
job_str = (f"{args.model_name[:8]}-{args.criterion}-L{args.num_layers}-N{args.d_model}-"
           f"H{args.num_heads}-M{args.dim_head}-G{args.qk_dim_head}-"
           f"F{args.num_sum_heads}-K{args.concat_pos_emb_dim}-"
           f"randomIP{args.use_random_first_projection}-"
           f"nores{args.no_residual}-nocls{args.no_cls_token}-"
           f"sinpos{args.use_sin_pos_enc}-concatpos{args.concat_pos_enc}-"
           f"randompos{args.use_random_position_encoding}-"
           f"learnAttn{args.learn_attention}-para{args.use_parallel}-"
           f"LearnedInput{args.add_learned_input_layer}-"
           f"binary{args.binary_class_choice}")

args.work_dir = os.path.join(
    args.work_dir, f"pretrained_{job_str}-{exp_hash}-{time.strftime('%Y%m%d-%H%M%S')}")
if not os.path.exists(args.work_dir):
    os.makedirs(args.work_dir)

work_dir_key = '/'.join(os.path.abspath(args.work_dir).split('/')[-3:])

# logging
log_file_name = f"{args.work_dir}/log.txt"
handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
logging.basicConfig(
    level=logging.INFO, format='%(message)s', handlers=handlers)

loginf = logging.info

loginf(f"Command executed: {sys.argv[:]}")
loginf(f"Args: {json.dumps(args.__dict__, indent=2)}")

info_string = (f"{work_dir_key}\nCommand executed: {sys.argv[:]}\n"
               f"Args: {json.dumps(args.__dict__, indent=2)}")

loginf(f"torch version: {torch.__version__}")
loginf(f"Work dir: {args.work_dir}")
loginf(f"Seed: {args.seed}")

# We will store a dictionary
dumped_file_path = os.path.join(args.work_dir, 'pretrained.pt')
args.loginf = loginf

model = get_model(args)

# load the model
loginf(f"Loading pre-trained model from {args.load_model_from}")
checkpoint = torch.load(args.load_model_from)
model.load_state_dict(checkpoint['model_state_dict'])

model = model.to('cuda')
# Go over the training dataset, compute and stores x_zeros and labels
# as a sanity check also compute the accuracy
loginf('Going through the training data...')
sum_mse_fn = nn.MSELoss(reduction='sum')
correct = 0
total_loss = 0
total = 0

all_labels_train = []
all_x_zeros_train = []

if args.min_num_training_points > 0:
    min_num_training_points = args.min_num_training_points
else:
    # use all points
    min_num_training_points = np.inf

if args.min_num_test_points > 0:
    min_num_test_points = args.min_num_test_points
else:
    # use all points
    min_num_test_points = np.inf

if args.min_num_extra_test_points > 0:
    min_num_extra_test_points = args.min_num_extra_test_points
else:
    # use all points
    min_num_extra_test_points = np.inf

# re-seed to get the same datapoints used for training
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

with torch.no_grad():
    for data in train_dl:
        inputs, target_labels = data
        inputs = inputs.to('cuda')  # (B, len, 1, 28, 28)
        target_labels = target_labels.to('cuda')  # (B, len)
        # bsz, slen = input_labels.shape

        inputs = inputs.transpose(0, 1)

        # convert labels to +1/-1
        target_labels = 2 * target_labels - 1.

        outputs, x_0 = model(
            inputs, get_x_zero=True,
            drop_heads=args.drop_heads)

        loss = sum_mse_fn(outputs, target_labels.float().unsqueeze(1))
        total_loss += loss.item()

        all_labels_train.append(target_labels.clone().to('cpu'))
        all_x_zeros_train.append(x_0.clone().to('cpu'))

        # decision based on sign (alternatively, check the distance to 1 and -1?)
        out_zeros = torch.zeros_like(outputs)
        outputs = torch.concat([out_zeros, outputs], dim=-1)
        zero_one_labels = outputs.argmax(-1)  # [B, 2]
        predicted = torch.where(zero_one_labels == 0, -1, 1)

        total += target_labels.size(0)
        correct += (predicted == target_labels).sum().item()

        if total > min_num_training_points:
            loginf(
                f'Minimum number of training points achieved: {min_num_training_points}')
            break
    acc = 100 * correct / total
    total_loss = total_loss / total
loginf(f'Training Accuracy: {acc} %')
loginf(f'Training Loss: {total_loss}')

# Now test set
all_labels_test = []
all_x_zeros_test = []

correct = 0
total_loss = 0
total = 0

with torch.no_grad():
    for data in test_dl:
        inputs, target_labels = data
        inputs = inputs.to('cuda')  # (B, len, 1, 28, 28)
        target_labels = target_labels.to('cuda')  # (B, len)

        inputs = inputs.transpose(0, 1)

        # convert labels to +1/-1
        target_labels = 2 * target_labels - 1.

        outputs, x_0 = model(
            inputs, get_x_zero=True, drop_heads=args.drop_heads)

        loss = sum_mse_fn(outputs, target_labels.float().unsqueeze(1))
        total_loss += loss.item()

        all_labels_test.append(target_labels.clone().to('cpu'))
        all_x_zeros_test.append(x_0.clone().to('cpu'))

        # decision based on sign (alternatively, check the distance to 1 and -1?)
        out_zeros = torch.zeros_like(outputs)
        outputs = torch.concat([out_zeros, outputs], dim=-1)
        zero_one_labels = outputs.argmax(-1)  # [B, 2]
        predicted = torch.where(zero_one_labels == 0, -1, 1)

        total += target_labels.size(0)
        correct += (predicted == target_labels).sum().item()
        if total > min_num_test_points:
            loginf(
                f'Minimum number of test points achieved: {min_num_test_points}')
            break
    acc = 100 * correct / total
    total_loss = total_loss / total
loginf(f'Test Accuracy: {acc} %')
loginf(f'Test Loss: {total_loss}')

# extract q/k and v weights for each layer
num_heads_eff = args.num_heads * args.num_sum_heads
w_q_list = []
w_k_list = []
w_v_list = []
for layer_id in range(args.num_layers):
    # q/k
    w_q = rearrange(model.transformer.layers[layer_id].to_q.weight,
                    '(h g) n0 -> h n0 g', h=num_heads_eff)
    w_q_list.append(w_q.clone())
    w_k = rearrange(model.transformer.layers[layer_id].to_k.weight,
                    '(h g) n0 -> h n0 g', h=num_heads_eff)
    w_k_list.append(w_k.clone())
    # v
    w_v = rearrange(model.transformer.layers[layer_id].to_v.weight,
                    '(h g) n -> h n g', h=num_heads_eff)
    w_v_list.append(w_v.clone())

# Finalize
# [P, N0]
x_zeros_train = torch.cat(all_x_zeros_train, dim=0).to('cpu')
x_zeros_test = torch.cat(all_x_zeros_test, dim=0).to('cpu')
# [P, 1]
y_labels_train = torch.cat(all_labels_train, dim=0).to('cpu')
y_labels_test = torch.cat(all_labels_test, dim=0).to('cpu')

# [L, F, N0, G]
w_q_weigts = torch.stack(w_q_list, dim=0).to('cpu')
w_k_weigts = torch.stack(w_k_list, dim=0).to('cpu')
w_v_weigts = torch.stack(w_v_list, dim=0).to('cpu')

# input projection and readout
w_input_proj = model.learned_input_layer.weight.to('cpu')
w_readout = model.readout_layer.weight.to('cpu')

if args.drop_heads is None:
    loginf(f'Writing to: {dumped_file_path}')

    torch.save({'x_zeros_train': x_zeros_train,
                'y_labels_train': y_labels_train,
                'x_zeros_test': x_zeros_test,
                'y_labels_test': y_labels_test,

                'w_q_weigts': w_q_weigts,
                'w_k_weigts': w_k_weigts,
                'w_v_weigts': w_v_weigts,

                'input_projection_weights': w_input_proj,
                'readout_weights': w_readout,

                'info_string': info_string},
                dumped_file_path)

loginf('Done.')
