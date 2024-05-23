from torchvision.utils import save_image

import argparse
import random

import torch
import numpy as np

from torchvision.transforms import (
    Compose, Resize, ToTensor, Normalize, Grayscale)

from torchmeta_local.utils.data import BatchMetaDataLoader

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

# parser.add_argument("--dump_train_set", action="store_true")

# extra trainig params
parser.add_argument("--limit_training_points", default=0, type=int)

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
args.num_workers = 4*args.gpus if args.gpus else 8
args.is_cls_token = True if not args.no_cls_token else False

if not args.gpus:
    args.precision = 32

# set in_context_learning
args.in_context_learning = True
args.criterion = 'binary_mse'

# Omniglot, extended to 3 channel, size 32x32
if args.use_grey_scale:
    args.in_c = 1
else:
    args.in_c = 3
args.size = 32

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
loginf("== In-Context Learning ==")

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

loginf(f"Dataset: {args.dataset}")
loginf(f"Greyscale: {args.use_grey_scale}")

args.dataset = "omniglot"

if args.dataset == "omniglot":
    from torchmeta_local.datasets.helpers import omniglot as data_cls
    norm_params = {'mean': [0.922], 'std': [0.084]}
    if args.use_grey_scale:
        transform = Compose(
            [Resize(32), Grayscale(num_output_channels=1),
             ToTensor(), Normalize(**norm_params)])
    else:
        transform = ToTensor()
else:
    assert args.dataset == "miniimagenet"
    from torchmeta_local.datasets.helpers import miniimagenet_32_norm_cache as data_cls
    # NB: miniimagenet only has 64/16/20 classes for train/valid/test
    # so for 2-way few-shot learning setting,
    # we only have 64!/(2!(64-2)!) = 2016 training sequences.
    transform = None  # the cached version is automatically 3-channel 32x32

# NB: hardcoded at the moment
n_way = 2
k_shot_train = 2
test_per_class = 1
num_samples_per_class = {'train': k_shot_train, 'test': test_per_class}

if args.augment_omniglot:
    from torchmeta_local.transforms import Rotation
    class_augmentations = [Rotation([90, 180, 270])]
else:
    class_augmentations = None

train_dataset = data_cls(
    args.data_dir, ways=2, shots=2,
    test_shots=2, meta_train=True,
    download=True, shuffle=True, seed=seed,
    num_samples_per_class=num_samples_per_class,
    transform=transform, class_augmentations=class_augmentations)
train_dl = BatchMetaDataLoader(
    train_dataset, batch_size=20, num_workers=1,
    pin_memory=True, drop_last=True)


count = 0
for data in train_dl:
    count += 1
    inputs, input_labels = data['train']
    print(inputs.shape)
    print(input_labels.shape)
    inputs = inputs[0]
    input_labels = input_labels[0]
    for i in range(len(input_labels)):
        print(f'label {input_labels[i]}, img_{count}_{i}_{input_labels[i]}')
        save_image(inputs[i], f'images_dumped/img_{count}_{i}_{input_labels[i]}.png')
    if count == 30:
       import sys; sys.exit(0)



