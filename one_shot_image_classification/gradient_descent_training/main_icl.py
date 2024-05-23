import argparse
import random

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from einops import repeat

import torchvision
from torchvision.transforms import (
    Compose, Resize, ToTensor, Lambda, Normalize, Grayscale)
import warmup_scheduler

import copy

from torchmeta_local.utils.data import BatchMetaDataLoader

from utils import get_model, get_criterion
from da import CutMix, MixUp

from icl_utils import get_icl_dataset

import sys
import os
import json
import logging
import hashlib
import time

from utils import get_experiment_name

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

if args.dataset == "omniglot":
    from torchmeta_local.datasets.helpers import omniglot_32_norm as data_cls
    norm_params = {'mean': [0.922], 'std': [0.084]}
    if args.use_grey_scale:
        transform = Compose(
            [Resize(32), Grayscale(num_output_channels=1),
             ToTensor(), Normalize(**norm_params)])
    else:
        transform = Compose(
            [Resize(32), ToTensor(), Normalize(**norm_params),
             Lambda(lambda x: x.repeat(3, 1, 1))])
else:
    assert args.dataset == "miniimagenet"
    from torchmeta_local.datasets.helpers import miniimagenet_32_norm_cache as data_cls
    # NB: miniimagenet only has 64/16/20 classes for train/valid/test
    # so for 2-way few-shot learning setting,
    # we only have 64!/(2!(64-2)!) = 2016 training sequences.
    transform = None  # the cached version is automatically 3-channel 32x32

# NB: hardcoded at the moment
n_way = 2
k_shot_train = args.icl_num_shots
test_per_class = 1
num_samples_per_class = {'train': k_shot_train, 'test': test_per_class}

if args.augment_omniglot:
    from torchmeta_local.transforms import Rotation
    class_augmentations = [Rotation([90, 180, 270])]
else:
    class_augmentations = None

train_dataset = data_cls(
    args.data_dir, ways=n_way, shots=k_shot_train,
    test_shots=test_per_class, meta_train=True,
    download=True, shuffle=True, seed=seed,
    num_samples_per_class=num_samples_per_class,
    transform=transform, class_augmentations=class_augmentations)
train_dl = BatchMetaDataLoader(
    train_dataset, batch_size=batch_size, num_workers=args.num_workers,
    pin_memory=True, drop_last=True)

val_dataset = data_cls(args.data_dir, ways=n_way, shots=k_shot_train,
                    test_shots=test_per_class, meta_val=True,
                    shuffle=True, seed=valid_seed, transform=transform)
if batch_size > len(val_dataset):
    val_batch_size = len(val_dataset)
else:
    val_batch_size = batch_size
valid_dl = BatchMetaDataLoader(
    val_dataset, batch_size=val_batch_size, num_workers=args.num_workers,
    pin_memory=True, drop_last=True)

test_dataset = data_cls(args.data_dir, ways=n_way, shots=k_shot_train,
                    test_shots=test_per_class, meta_test=True,
                    shuffle=True, seed=test_seed, transform=transform)
if batch_size > len(test_dataset):
    test_batch_size = len(test_dataset)
else:
    test_batch_size = batch_size
test_dl = BatchMetaDataLoader(
    test_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
    pin_memory=True, drop_last=True)


# create a new dataloader with limited datapoints.
if args.limit_training_points:
    limited_dataset = []
    if args.limit_training_points <= args.batch_size:
        limit_batch_number = 1
    elif args.limit_training_points % args.batch_size == 0:
        limit_batch_number = args.limit_training_points // args.batch_size
    else:
        limit_batch_number = args.limit_training_points // args.batch_size + 1
    loginf(f'Requested {args.limit_training_points} training samples')
    loginf(f'Using {limit_batch_number * args.batch_size} samples'
                f'({limit_batch_number} batches of size {args.batch_size})')

    from torch.utils.data import Dataset, DataLoader
    class ICLLimitedDataset(Dataset):
        def __init__(self, train_dl, limit_batch_number, equal_classes=False):
            self.dataset = []
            self.sample_dataset(train_dl, limit_batch_number)
            self.equal_classes = equal_classes

        def sample_dataset(self, train_dl, limit_batch_number):
            if not self.equal_classes:
                for k, data in enumerate(train_dl):
                    if k > limit_batch_number:
                        break
                    inputs, input_labels = data['train']
                    test_inputs, test_input_labels = data['test']
                    bsz = inputs.shape[0]
                    for i in range(bsz):
                        datapoint = [
                            [inputs[i], input_labels[i]],
                            [test_inputs[i], test_input_labels[i]]]
                        self.dataset.append(datapoint)
            else:
                assert args.limit_training_points % 2 == 0
                # sample one extra batch
                limit_batch_number += 1
                class_0_counter = 0
                class_1_counter = 0
                for k, data in enumerate(train_dl):
                    inputs, input_labels = data['train']
                    test_inputs, test_input_labels = data['test']
                    bsz = inputs.shape[0]
                    for i in range(bsz):
                        datapoint = [
                            [inputs[i], input_labels[i]],
                            [test_inputs[i], test_input_labels[i]]]
                        if test_input_labels[i] == 0:
                            if class_0_counter < args.limit_training_points:
                                
                        self.dataset.append(datapoint)

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            return self.dataset[idx]
        
    train_dataset = ICLLimitedDataset(
        train_dl, limit_batch_number, equal_classes=args.equal_class_balance)
    train_dl = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

# Get test MNIST and CIFAR-10 dataset
# no train/validation separation here (could be introduced for meta-validation)
_icl_input_labels = torch.tensor([0, 1, 2], device='cuda').unsqueeze(-1)  # fixed order; does not matter for testing
mnist_dataloader = get_icl_dataset("mnist",
    args.data_dir, args.eval_batch_size, args.num_workers,
    args.binary_class_choice, grey_scale=args.use_grey_scale)

# FashionM
fmnist_dataloader = get_icl_dataset("fmnist",
    args.data_dir, args.eval_batch_size, args.num_workers,
    args.binary_class_choice, grey_scale=args.use_grey_scale)

# Prepare CIFAR-10 dataset
cifar10_dataloader = get_icl_dataset("cifar10",
    args.data_dir, args.eval_batch_size, args.num_workers,
    args.binary_class_choice, grey_scale=args.use_grey_scale)

extra_dataloaders = {
    "MNIST": mnist_dataloader, "FMNIST": fmnist_dataloader,
    "CIFAR10": cifar10_dataloader}


class InContextNet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        # self.hparams = hparams
        self.hparams.update(vars(args))
        self.model = get_model(args)
        self.criterion = get_criterion(args)
        self.type_criterion = args.criterion
        self.num_classes = args.num_classes
        self.report_every = args.report_every
        self.validate_every = args.validate_every
        self.use_wandb = args.use_wandb
        self.no_lr_scheduler = args.no_lr_scheduler

        if args.init_model_from is not None:
            loginf(f"Init key/query matrices from: {args.init_model_from}")
            checkpoint = torch.load(args.init_model_from)
            checkpoint_dict = checkpoint['model_state_dict']
            model_dict = self.model.state_dict()
            new_dict = {}
            if args.init_entire_model:
                loginf(f"loading the entire model")
                self.model.load_state_dict(checkpoint_dict)
            else:
            # we assume that the dim of the old/checkpoint model is bigger.
                for key, value in checkpoint_dict.items():  # 2-dim
                    if 'to_q.weight' in key:
                        loginf(f"loading: {key}")
                        new_dict[key] = value
                    elif 'to_k.weight' in key:
                        loginf(f"loading: {key}")
                        new_dict[key] = value
                    elif 'label_embedding' in key:
                        loginf(f"loading: {key}")
                        new_dict[key] = value
                    elif 'to_patch_embedding' in key:
                        loginf(f"loading: {key}")
                        new_dict[key] = value
                    elif 'to_v.weight' in key:
                        if args.init_also_value:
                            loginf(f"loading: {key}")
                            new_dict[key] = value
                    elif args.init_also_readout:
                        if 'readout_layer' in key:
                            loginf(f"loading: {key}")
                            new_dict[key] = value
                model_dict.update(new_dict)
                self.model.load_state_dict(model_dict)

        if args.save_init_model:
            torch.save({'model_state_dict': self.model.state_dict()},
                       args.init_model_path)

        if args.cutmix:
            self.cutmix = CutMix(args.size, beta=1.)
        if args.mixup:
            self.mixup = MixUp(alpha=1.)

        self.best_val_acc = 0.
        self.best_checkpoint = None
        self.last_checkpoint = None

        self.best_model_path = args.best_model_path
        self.latest_model_path = args.latest_model_path

        self.loginf = args.loginf

        self.use_limited_training_points = (
            True if args.limit_training_points > 0 else False)

        self.icl_eval_num_iter = args.icl_eval_num_iter

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            weight_decay=self.hparams.weight_decay)
        if self.no_lr_scheduler:
            return [self.optimizer]
        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.hparams.max_epochs,
            eta_min=self.hparams.min_lr)
        self.scheduler = warmup_scheduler.GradualWarmupScheduler(
            self.optimizer, multiplier=1.,
            total_epoch=self.hparams.warmup_epoch,
            after_scheduler=self.base_scheduler)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        if self.use_limited_training_points:
            inputs = batch[0][0]
            input_labels = batch[0][1]
            test_inputs = batch[1][0]
            test_input_labels = batch[1][1]
        else:
            inputs, input_labels = batch['train']
            test_inputs, test_input_labels = batch['test']

        inputs = inputs.to('cuda')  # (B, len, 1, 28, 28)
        input_labels = input_labels.to('cuda')  # (B, len)

        inputs = inputs.transpose(0, 1)
        input_labels = input_labels.transpose(0, 1)

        test_inputs = test_inputs[:, 0].to('cuda').unsqueeze(0)  # (s=1, b, *)
        test_input_labels = test_input_labels[:, 0].to('cuda')  # (s=1, b)

        unk_labels = torch.zeros_like(test_input_labels.unsqueeze(0))
        unk_labels = unk_labels + 2

        net_input = torch.cat([inputs, test_inputs], dim=0)
        net_input_label = torch.cat([input_labels, unk_labels], dim=0)

        out = self.model(net_input, icl_labels=net_input_label)

        # convert labels to +1/-1
        target_labels = 2 * test_input_labels - 1.

        loss = self.criterion(out, target_labels.unsqueeze(-1))

        if self.type_criterion == 'binary_mse':
            # decision based on sign (alternatively, check the distance to 1 and -1?)
            out_zeros = torch.zeros_like(out)
            out = torch.concat([out_zeros, out], dim=-1)
            zero_one_labels = out.argmax(-1)  # [B, 2]
            out = torch.where(zero_one_labels == 0, -1, 1)
            acc = torch.eq(out, target_labels).float().mean()
        else:
            assert False, "Not supported yet"
            acc = torch.eq(out.argmax(-1), test_input_labels).float().mean()

        self.log("loss", loss)
        self.log("acc", acc)
        if self.report_every > 0 and (batch_idx % self.report_every == 0):
            self.loginf(f'[epoch {self.current_epoch}, step {batch_idx}] Train batch loss: {loss}')
            self.loginf(f'[epoch {self.current_epoch}, step {batch_idx}] Train batch acc {100 * acc} %')
            if self.use_wandb:
                wandb.log({"train_batch_loss": loss,
                        "train_batch_acc": 100 * acc,})
            if not self.no_lr_scheduler:
                self.scheduler.step()
            self.loginf(f'[epoch {self.current_epoch}, step {batch_idx}] Learning rate: {self.optimizer.param_groups[0]["lr"]}')
            if self.use_wandb:
                wandb.log({"learning_rate": self.optimizer.param_groups[0]["lr"]})

        if self.validate_every > 0 and (batch_idx % self.validate_every == 0):
            self.training_epoch_end(step=batch_idx)
        return loss

    def training_epoch_end(self, outputs=None, step=None):
        self.log("lr", self.optimizer.param_groups[0]["lr"], on_epoch=self.current_epoch)
        self.loginf(f'[epoch {self.current_epoch}] Learning rate: {self.optimizer.param_groups[0]["lr"]}')
        if self.use_wandb:
            wandb.log({"learning_rate": self.optimizer.param_groups[0]["lr"]})

        correct = 0
        total = 0
        iter = 0
        self.model.eval()
        with torch.no_grad():
            for data in valid_dl:
                inputs, input_labels = data['train']
                inputs = inputs.to('cuda')  # (B, len, 1, 28, 28)
                input_labels = input_labels.to('cuda')  # (B, len)

                inputs = inputs.transpose(0, 1)
                input_labels = input_labels.transpose(0, 1)

                test_inputs, test_input_labels = data['test']
                # already shuffled order, just take the first one
                test_inputs = test_inputs[:, 0].to('cuda').unsqueeze(0)  # (s=1, b, *)
                test_input_labels = test_input_labels[:, 0].to('cuda')  # (s=1, b)

                unk_labels = torch.zeros_like(test_input_labels.unsqueeze(0))
                # hardcoded for binary classification `2 = num_classes`
                unk_labels = unk_labels + 2

                # convert labels to +1/-1
                target_labels = 2 * test_input_labels - 1.

                net_input = torch.cat([inputs, test_inputs], dim=0)
                net_input_label = torch.cat([input_labels, unk_labels], dim=0)

                outputs = self.model(net_input, icl_labels=net_input_label)

                if self.type_criterion == 'binary_mse':
                    # decision based on sign (alternatively, check the distance to 1 and -1?)
                    out_zeros = torch.zeros_like(outputs)
                    outputs = torch.concat([out_zeros, outputs], dim=-1)
                    zero_one_labels = outputs.argmax(-1)
                    predicted = torch.where(zero_one_labels == 0, -1, 1)
                else:
                    assert False
                    _, predicted = outputs.max(1)

                # _, predicted = outputs.max(1)
                total += target_labels.size(0)
                correct += (predicted == target_labels).sum().item()
                iter += 1
                if iter > self.icl_eval_num_iter:
                    break
            val_acc = 100 * correct / total
        self.loginf(f'[epoch {self.current_epoch}, step {step}] Validation Accuracy: {val_acc} %')

        if self.use_wandb:
            wandb.log({"val_acc": val_acc})

        if self.best_val_acc < val_acc:
            self.best_val_acc = val_acc
            self.loginf(
                f'[epoch {self.current_epoch}, step {step}] Current best validation accuracy: {val_acc} %')
            self.loginf('saving the best checkpoint')
            torch.save({'epoch': self.current_epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'valid_acc': val_acc}, self.best_model_path)
            self.best_checkpoint = copy.deepcopy(self.model).state_dict()

        torch.save({'epoch': self.current_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'valid_acc': val_acc}, self.latest_model_path)
        self.last_checkpoint = copy.deepcopy(self.model).state_dict()

        with torch.no_grad():
            for key in extra_dataloaders.keys():
                correct = 0
                total = 0
                for data in extra_dataloaders[key]:
                    inputs, target_labels = data
                    inputs = inputs.to('cuda')  # (B, len, 1, 28, 28)
                    target_labels = target_labels.to('cuda')  # (B, len)
                    bsz = inputs.shape[0]

                    inputs = inputs.transpose(0, 1)
                    input_labels = repeat(_icl_input_labels, 't 1 -> t b', b = bsz)

                    outputs = self.model(inputs, icl_labels=input_labels)

                    if self.type_criterion == 'binary_mse':
                        # decision based on sign (alternatively, check the distance to 1 and -1?)
                        out_zeros = torch.zeros_like(outputs)
                        outputs = torch.concat([out_zeros, outputs], dim=-1)
                        zero_one_labels = outputs.argmax(-1)
                        predicted = torch.where(zero_one_labels == 0, -1, 1)
                    else:
                        assert False
                        _, predicted = outputs.max(1)

                    total += target_labels.size(0)
                    correct += (predicted == target_labels).sum().item()
                val_acc = 100 * correct / total
                self.loginf(f'[epoch {self.current_epoch}, step {step}] {key} Test Accuracy: {val_acc} %')

                if self.use_wandb:
                    wandb.log({f"{key}_acc": val_acc})

        self.model.train()

    def validation_step(self, batch, batch_idx):
        return None

    def test_step(self, batch, batch_idx):
        return None

    def test_epoch_end(self, outputs):
        # last checkpoint
        self.loginf('=================== Final Eval ======================== ')
        self.loginf('Test, loading last checkpoint')
        assert self.last_checkpoint is not None
        self.model.load_state_dict(self.last_checkpoint)
        correct = 0
        total = 0
        iter = 0
        self.model.eval()
        with torch.no_grad():
            for data in test_dl:

                inputs, input_labels = data['train']
                inputs = inputs.to('cuda')  # (B, len, 1, 28, 28)
                input_labels = input_labels.to('cuda')  # (B, len)
                # bsz, slen = input_labels.shape

                inputs = inputs.transpose(0, 1)
                input_labels = input_labels.transpose(0, 1)

                test_inputs, test_input_labels = data['test']
                # already shuffled order, just take the first one
                test_inputs = test_inputs[:, 0].to('cuda').unsqueeze(0)  # (s=1, b, *)
                test_input_labels = test_input_labels[:, 0].to('cuda')  # (s=1, b)

                unk_labels = torch.zeros_like(test_input_labels.unsqueeze(0))
                # hardcoded for binary classification `2 = num_classes`
                unk_labels = unk_labels + 2

                # convert labels to +1/-1
                target_labels = 2 * test_input_labels - 1.

                net_input = torch.cat([inputs, test_inputs], dim=0)
                net_input_label = torch.cat([input_labels, unk_labels], dim=0)

                outputs = self.model(net_input, icl_labels=net_input_label)

                if self.type_criterion == 'binary_mse':
                    # decision based on sign (alternatively, check the distance to 1 and -1?)
                    out_zeros = torch.zeros_like(outputs)
                    outputs = torch.concat([out_zeros, outputs], dim=-1)
                    zero_one_labels = outputs.argmax(-1)  # [B, 2]
                    predicted = torch.where(zero_one_labels == 0, -1, 1)
                else:
                    assert False
                    _, predicted = outputs.max(1)
                total += target_labels.size(0)
                correct += (predicted == target_labels).sum().item()
                if iter > self.icl_eval_num_iter:
                    break
            val_acc = 100 * correct / total
        self.loginf(f'[last model] Test Accuracy: {val_acc} %')

        if args.binary_task_only_two_classes or args.binary_task or args.in_context_learning:
            assert self.type_criterion == 'binary_mse'
            self.loginf('Evaluate training loss...')
            sum_mse_fn = nn.MSELoss(reduction='sum')
            correct = 0
            total_loss = 0
            total = 0
            iter = 0
            with torch.no_grad():
                for data in train_dl:
                    if self.use_limited_training_points:
                        inputs = data[0][0]
                        input_labels = data[0][1]
                        test_inputs = data[1][0]
                        test_input_labels = data[1][1]
                    else:
                        inputs, input_labels = data['train']
                        test_inputs, test_input_labels = data['test']

                    inputs = inputs.to('cuda')  # (B, len, 1, 28, 28)
                    input_labels = input_labels.to('cuda')  # (B, len)

                    inputs = inputs.transpose(0, 1)
                    input_labels = input_labels.transpose(0, 1)

                    # test_inputs, test_input_labels = data['test']
                    # already shuffled order, just take the first one
                    test_inputs = test_inputs[:, 0].to('cuda').unsqueeze(0)  # (s=1, b, *)
                    test_input_labels = test_input_labels[:, 0].to('cuda')  # (s=1, b)

                    unk_labels = torch.zeros_like(test_input_labels.unsqueeze(0))
                    # hardcoded for binary classification `2 = num_classes`
                    unk_labels = unk_labels + 2

                    # convert labels to +1/-1
                    target_labels = 2 * test_input_labels - 1.

                    net_input = torch.cat([inputs, test_inputs], dim=0)
                    net_input_label = torch.cat([input_labels, unk_labels], dim=0)

                    outputs = self.model(net_input, icl_labels=net_input_label)

                    # outputs = self.model(images)
                    loss = sum_mse_fn(outputs, target_labels.float().unsqueeze(1))
                    total_loss += loss.item()

                    # decision based on sign (alternatively, check the distance to 1 and -1?)
                    out_zeros = torch.zeros_like(outputs)
                    outputs = torch.concat([out_zeros, outputs], dim=-1)
                    zero_one_labels = outputs.argmax(-1)  # [B, 2]
                    predicted = torch.where(zero_one_labels == 0, -1, 1)

                    total += target_labels.size(0)
                    correct += (predicted == target_labels).sum().item()
                    if iter > self.icl_eval_num_iter:
                        break
                acc = 100 * correct / total
                total_loss = total_loss / total
            self.loginf(f'[last model] Training Accuracy: {acc} %')
            self.loginf(f'[last model] Training Loss: {total_loss}')

        with torch.no_grad():
            for key in extra_dataloaders.keys():
                correct = 0
                total = 0
                for data in extra_dataloaders[key]:
                    inputs, target_labels = data
                    inputs = inputs.to('cuda')  # (B, len, 1, 28, 28)
                    target_labels = target_labels.to('cuda')  # (B, len)
                    bsz = inputs.shape[0]

                    inputs = inputs.transpose(0, 1)
                    input_labels = repeat(_icl_input_labels, 't 1 -> t b', b = bsz)

                    outputs = self.model(inputs, icl_labels=input_labels)

                    if self.type_criterion == 'binary_mse':
                        # decision based on sign (alternatively, check the distance to 1 and -1?)
                        out_zeros = torch.zeros_like(outputs)
                        outputs = torch.concat([out_zeros, outputs], dim=-1)
                        zero_one_labels = outputs.argmax(-1)
                        predicted = torch.where(zero_one_labels == 0, -1, 1)
                    else:
                        assert False
                        _, predicted = outputs.max(1)

                    total += target_labels.size(0)
                    correct += (predicted == target_labels).sum().item()
                val_acc = 100 * correct / total
                self.loginf(f'[last model] {key} Test Accuracy: {val_acc} %')

        # Best 
        self.loginf('Test, loading best checkpoint')
        assert self.best_checkpoint is not None
        self.model.load_state_dict(self.best_checkpoint)
        correct = 0
        total = 0
        iter = 0
        self.model.eval()
        with torch.no_grad():
            for data in test_dl:
                inputs, input_labels = data['train']
                inputs = inputs.to('cuda')  # (B, len, 1, 28, 28)
                input_labels = input_labels.to('cuda')  # (B, len)
                # bsz, slen = input_labels.shape

                inputs = inputs.transpose(0, 1)
                input_labels = input_labels.transpose(0, 1)

                test_inputs, test_input_labels = data['test']
                # already shuffled order, just take the first one
                test_inputs = test_inputs[:, 0].to('cuda').unsqueeze(0)  # (s=1, b, *)
                test_input_labels = test_input_labels[:, 0].to('cuda')  # (s=1, b)

                unk_labels = torch.zeros_like(test_input_labels.unsqueeze(0))
                # hardcoded for binary classification `2 = num_classes`
                unk_labels = unk_labels + 2

                # convert labels to +1/-1
                target_labels = 2 * test_input_labels - 1.

                net_input = torch.cat([inputs, test_inputs], dim=0)
                net_input_label = torch.cat([input_labels, unk_labels], dim=0)

                outputs = self.model(net_input, icl_labels=net_input_label)

                if self.type_criterion == 'binary_mse':
                    # decision based on sign (alternatively, check the distance to 1 and -1?)
                    out_zeros = torch.zeros_like(outputs)
                    outputs = torch.concat([out_zeros, outputs], dim=-1)
                    zero_one_labels = outputs.argmax(-1)  # [B, 2]
                    predicted = torch.where(zero_one_labels == 0, -1, 1)
                else:
                    _, predicted = outputs.max(1)
                total += target_labels.size(0)
                correct += (predicted == target_labels).sum().item()
                if iter > self.icl_eval_num_iter:
                    break
            val_acc = 100 * correct / total
        self.loginf(f'[best model] Test Accuracy: {val_acc} %')

        with torch.no_grad():
            for key in extra_dataloaders.keys():
                correct = 0
                total = 0
                for data in extra_dataloaders[key]:
                    inputs, target_labels = data
                    inputs = inputs.to('cuda')  # (B, len, 1, 28, 28)
                    target_labels = target_labels.to('cuda')  # (B, len)
                    bsz = inputs.shape[0]

                    inputs = inputs.transpose(0, 1)
                    input_labels = repeat(_icl_input_labels, 't 1 -> t b', b = bsz)

                    outputs = self.model(inputs, icl_labels=input_labels)

                    if self.type_criterion == 'binary_mse':
                        # decision based on sign (alternatively, check the distance to 1 and -1?)
                        out_zeros = torch.zeros_like(outputs)
                        outputs = torch.concat([out_zeros, outputs], dim=-1)
                        zero_one_labels = outputs.argmax(-1)
                        predicted = torch.where(zero_one_labels == 0, -1, 1)
                    else:
                        assert False
                        _, predicted = outputs.max(1)

                    total += target_labels.size(0)
                    correct += (predicted == target_labels).sum().item()
                val_acc = 100 * correct / total
                self.loginf(f'[best model] {key} Test Accuracy: {val_acc} %')

        if args.binary_task_only_two_classes or args.binary_task or args.in_context_learning:
            assert self.type_criterion == 'binary_mse'
            self.loginf('Evaluate training loss...')
            sum_mse_fn = nn.MSELoss(reduction='sum')
            correct = 0
            total_loss = 0
            total = 0
            iter = 0
            with torch.no_grad():
                for data in train_dl:
                    if self.use_limited_training_points:
                        inputs = data[0][0]
                        input_labels = data[0][1]
                        test_inputs = data[1][0]
                        test_input_labels = data[1][1]
                    else:
                        inputs, input_labels = data['train']
                        test_inputs, test_input_labels = data['test']

                    inputs = inputs.to('cuda')  # (B, len, 1, 28, 28)
                    input_labels = input_labels.to('cuda')  # (B, len)
                    # bsz, slen = input_labels.shape

                    inputs = inputs.transpose(0, 1)
                    input_labels = input_labels.transpose(0, 1)

                    # test_inputs, test_input_labels = data['test']
                    # already shuffled order, just take the first one
                    test_inputs = test_inputs[:, 0].to('cuda').unsqueeze(0)  # (s=1, b, *)
                    test_input_labels = test_input_labels[:, 0].to('cuda')  # (s=1, b)

                    unk_labels = torch.zeros_like(test_input_labels.unsqueeze(0))
                    # hardcoded for binary classification `2 = num_classes`
                    unk_labels = unk_labels + 2

                    # convert labels to +1/-1
                    target_labels = 2 * test_input_labels - 1.

                    net_input = torch.cat([inputs, test_inputs], dim=0)
                    net_input_label = torch.cat([input_labels, unk_labels], dim=0)

                    outputs = self.model(net_input, icl_labels=net_input_label)

                    loss = sum_mse_fn(outputs, target_labels.float().unsqueeze(1))
                    total_loss += loss.item()

                    # decision based on sign (alternatively, check the distance to 1 and -1?)
                    out_zeros = torch.zeros_like(outputs)
                    outputs = torch.concat([out_zeros, outputs], dim=-1)
                    zero_one_labels = outputs.argmax(-1)  # [B, 2]
                    predicted = torch.where(zero_one_labels == 0, -1, 1)

                    total += target_labels.size(0)
                    correct += (predicted == target_labels).sum().item()
                    if iter > self.icl_eval_num_iter:
                        break
                acc = 100 * correct / total
                total_loss = total_loss / total
            self.loginf(f'[best model] Training Accuracy: {acc} %')
            self.loginf(f'[best model] Training Loss: {total_loss}')
            self.loginf('======== END ====================================== ')

    def _log_image(self, image):
        grid = torchvision.utils.make_grid(image, nrow=4)
        self.logger.experiment.log_image(grid.permute(1,2,0))
        print("[INFO] LOG IMAGE!!!")


# wandb settings
if args.use_wandb:  # configure wandb.
    import wandb
    use_wandb = True
    # fix to remove extra HTTPS connection logs
    # https://stackoverflow.com/questions/11029717/how-do-i-disable-log-messages-from-the-requests-library
    logging.getLogger("requests").setLevel(logging.WARNING)

    if args.project_name is None:
        project_name = (os.uname()[1]
                        + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        project_name = args.project_name

    wandb.init(
        project=project_name, settings=wandb.Settings(start_method='fork'))
    # or `settings=wandb.Settings(start_method='thread')`
    if args.job_name is None:
        wandb.run.name = f"{os.uname()[1]}//" \
                         f"{job_str}" \
                         f"//PATH'{work_dir_key}'//"
    else:
        wandb.run.name = f"{os.uname()[1]}//{args.job_name}"

    config = wandb.config
    config.host = os.uname()[1]  # host node name
    config.seed = args.seed
    config.show_progress_bar = args.show_progress_bar
    config.dataset = args.dataset
    config.num_classes = args.num_classes
    config.model_name = args.model_name
    config.patch_size = args.patch_size
    config.batch_size = args.batch_size
    config.eval_batch_size = args.eval_batch_size
    config.lr = args.lr
    config.min_lr = args.min_lr
    config.beta1 = args.beta1
    config.beta2 = args.beta2
    config.off_benchmark = args.off_benchmark
    config.max_epochs = args.max_epochs
    config.dry_run = args.dry_run
    config.weight_decay = args.weight_decay
    config.warmup_epoch = args.warmup_epoch
    config.precision = args.precision
    config.gradient_clip = args.gradient_clip
    config.report_every = args.report_every
    config.criterion = args.criterion
    config.label_smoothing = args.label_smoothing
    config.smoothing = args.smoothing
    config.rcpaste = args.rcpaste
    config.cutmix = args.cutmix
    config.mixup = args.mixup
    config.autoaugment = args.autoaugment
    config.num_layers = args.num_layers
    config.d_model = args.d_model
    config.num_heads = args.num_heads
    config.dim_head = args.dim_head
    config.qk_dim_head = args.qk_dim_head
    config.num_sum_heads = args.num_sum_heads
    config.concat_pos_emb_dim = args.concat_pos_emb_dim
    config.no_residual = args.no_residual
    config.no_cls_token = args.no_cls_token
    config.use_random_first_projection = args.use_random_first_projection
    config.use_sin_pos_enc = args.use_sin_pos_enc
    config.use_random_position_encoding = args.use_random_position_encoding
    config.concat_pos_enc = args.concat_pos_enc
    config.remove_diag_scale = args.remove_diag_scale
    config.d_ff = args.d_ff
    config.dropout = args.dropout
    config.equal_head_dim_model_dim = args.equal_head_dim_model_dim
    config.low_rank_qk = args.low_rank_qk
    config.use_parallel = args.use_parallel
    config.add_learned_input_layer = args.add_learned_input_layer
    config.binary_class_choice = args.binary_class_choice
    config.learn_attention = args.learn_attention
    config.readout_type = args.readout_type
    config.vit_no_feedforward = args.vit_no_feedforward
    config.additive_icl_label_embedding = args.additive_icl_label_embedding
    config.augment_omniglot = args.augment_omniglot
    config.use_grey_scale = args.use_grey_scale
    config.freeze_value = args.freeze_value
else:
    use_wandb = False


args.best_model_path = os.path.join(args.work_dir, 'best_model.pt')
args.latest_model_path = os.path.join(args.work_dir, 'latest_model.pt')
args.init_model_path = os.path.join(args.work_dir, 'init_model.pt')

args.loginf = loginf

experiment_name = get_experiment_name(args)
loginf(experiment_name)

loginf("[INFO] Log with CSV")
logger = pl.loggers.CSVLogger(
    save_dir="logs",
    name=experiment_name
)

if args.show_progress_bar:
    refresh_rate = 1
else:
    refresh_rate = 0

net = InContextNet(args)

trainer = pl.Trainer(
    precision=args.precision,
    fast_dev_run=args.dry_run,
    gpus=args.gpus,
    benchmark=args.benchmark,
    logger=logger,
    max_epochs=args.max_epochs,
    weights_summary="full",
    progress_bar_refresh_rate=refresh_rate,
    gradient_clip_val=args.gradient_clip)

trainer.fit(model=net, train_dataloader=train_dl, val_dataloaders=valid_dl)
trainer.test(model=net, test_dataloaders=test_dl)
