import argparse

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

import torchvision
import warmup_scheduler

import copy

from utils import get_model, get_criterion
from da import CutMix, MixUp

import sys
import os
import json
import logging
import hashlib
import time

from utils import get_dataset, get_experiment_name

parser = argparse.ArgumentParser()
parser.add_argument("--show_progress_bar", action="store_true")
parser.add_argument("--dataset", default="c10", type=str,
                    help="[c10, c100, svhn]")
parser.add_argument("--valid_set_size", default=1000, type=int)
parser.add_argument("--num_classes", default=10, type=int)
parser.add_argument("--model_name", default="my_vit", type=str)
parser.add_argument("--patch_size", default=4, type=int)
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

args.in_context_learning = False

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)

args.benchmark = True if not args.off_benchmark else False
args.gpus = torch.cuda.device_count()
args.num_workers = 4*args.gpus if args.gpus else 8
args.is_cls_token = True if not args.no_cls_token else False

if not args.gpus:
    args.precision = 32

train_ds, test_ds = get_dataset(args)

# create proper validation set
# hard coded for cifar-10
num_train_data_points = train_ds.__len__()
valid_set_size = args.valid_set_size
idx = np.arange(num_train_data_points)
val_indices = idx[num_train_data_points-valid_set_size:]
train_indices= idx[:-valid_set_size]

batch_size = args.batch_size


if args.binary_task:
    if args.criterion != 'binary_mse':
        # force set criterion; put a loginf here.
        args.criterion = 'binary_mse'
        args.num_classes = 1  # single output

    # train
    tmp_targets = torch.ByteTensor(train_ds.targets)
    idx_0 = tmp_targets < 5
    idx_1 = tmp_targets > 4

    targets_0 = (tmp_targets[idx_0].int() - tmp_targets[idx_0].int() + 1).tolist()
    targets_1 = (tmp_targets[idx_1].int() - tmp_targets[idx_1].int() - 1).tolist()

    # print(targets_0)
    # print(targets_1)
    # print(tmp_targets.dtype)

    # import sys; sys.exit(0)

    valid_set_size = valid_set_size // 2

    val_indices = np.concatenate(
        (idx[:len(targets_0)][-valid_set_size:],
         idx[-valid_set_size:]))
    
    train_indices = np.concatenate(
        (idx[:len(targets_0)-valid_set_size],
         idx[len(targets_0):num_train_data_points-valid_set_size]))

    train_ds.targets = targets_0 + targets_1  # list concatenation

    train_ds.data = np.concatenate(
        (train_ds.data[idx_0], train_ds.data[idx_1]))

    # test
    tmp_targets = torch.ByteTensor(test_ds.targets)
    idx_0 = tmp_targets < 5
    idx_1 = tmp_targets > 4

    targets_0 = (tmp_targets[idx_0].int() - tmp_targets[idx_0].int() + 1).tolist()
    targets_1 = (tmp_targets[idx_1].int() - tmp_targets[idx_1].int() - 1).tolist()

    test_ds.targets = targets_0 + targets_1  # list concatenation
    test_ds.data = np.concatenate(
        (test_ds.data[idx_0], test_ds.data[idx_1]))

if args.binary_task_only_two_classes:
    if args.criterion != 'binary_mse':
        # force set criterion; put a loginf here.
        args.criterion = 'binary_mse'
        args.num_classes = 1  # single output

    # train
    class_list = list(args.binary_class_choice)
    class_0 = int(class_list[0])
    class_1 = int(class_list[1])
    assert 0 <= class_0 <= 9
    assert 0 <= class_1 <= 9

    tmp_targets = torch.ByteTensor(train_ds.targets)
    idx_0 = tmp_targets == class_0
    idx_1 = tmp_targets == class_1

    targets_0 = (tmp_targets[idx_0].int() - tmp_targets[idx_0].int() + 1).tolist()
    targets_1 = (tmp_targets[idx_1].int() - tmp_targets[idx_1].int() - 1).tolist()

    valid_set_size = valid_set_size // 2

    idx = np.arange(len(targets_0) + len(targets_1))

    val_indices = np.concatenate(
        (idx[:len(targets_0)][-valid_set_size:],
         idx[-valid_set_size:]))
    
    train_indices = np.concatenate(
        (idx[:len(targets_0)-valid_set_size],
         idx[len(targets_0):len(idx)-valid_set_size]))

    train_ds.targets = targets_0 + targets_1  # list concatenation

    train_ds.data = np.concatenate(
        (train_ds.data[idx_0], train_ds.data[idx_1]))

    # test
    tmp_targets = torch.ByteTensor(test_ds.targets)
    idx_0 = tmp_targets == class_0
    idx_1 = tmp_targets == class_1

    targets_0 = (tmp_targets[idx_0].int() - tmp_targets[idx_0].int() + 1).tolist()
    targets_1 = (tmp_targets[idx_1].int() - tmp_targets[idx_1].int() - 1).tolist()

    test_ds.targets = targets_0 + targets_1  # list concatenation
    test_ds.data = np.concatenate(
        (test_ds.data[idx_0], test_ds.data[idx_1]))

if args.limit_training_points > 0:
    np.random.shuffle(train_indices)
    train_indices = train_indices[:args.limit_training_points]

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

train_dl = torch.utils.data.DataLoader(
    train_ds, batch_size=batch_size, sampler=train_sampler,
    num_workers=args.num_workers, pin_memory=True)

valid_dl = torch.utils.data.DataLoader(
    train_ds, batch_size=batch_size, sampler=valid_sampler,
    num_workers=args.num_workers, pin_memory=True)

test_dl = torch.utils.data.DataLoader(
    test_ds, batch_size=args.eval_batch_size, num_workers=args.num_workers,
    pin_memory=True)


# if __name__ == "__main__":

class Net(pl.LightningModule):
    def __init__(self, args):
        super(Net, self).__init__()
        # self.hparams = hparams
        self.hparams.update(vars(args))
        self.model = get_model(args)
        self.criterion = get_criterion(args)
        self.type_criterion = args.criterion
        self.num_classes = args.num_classes
        self.report_every = args.report_every
        self.use_wandb = args.use_wandb

        if args.cutmix:
            self.cutmix = CutMix(args.size, beta=1.)
        if args.mixup:
            self.mixup = MixUp(alpha=1.)
        # self.log_image_flag = hparams.api_key is None

        self.best_val_acc = 0.
        self.best_checkpoint = None
        self.last_checkpoint = None

        self.best_model_path = args.best_model_path
        self.latest_model_path = args.latest_model_path

        self.loginf = args.loginf

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            weight_decay=self.hparams.weight_decay)
        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.hparams.max_epochs,
            eta_min=self.hparams.min_lr)
        self.scheduler = warmup_scheduler.GradualWarmupScheduler(
            self.optimizer, multiplier=1.,
            total_epoch=self.hparams.warmup_epoch,
            after_scheduler=self.base_scheduler)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        img, label = batch
        if self.hparams.cutmix or self.hparams.mixup:
            assert False
        else:
            out = self(img)
            if self.type_criterion == 'mse':
                loss_label = torch.nn.functional.one_hot(
                    label, num_classes=self.num_classes).float()
            elif self.type_criterion == 'sigm_mse':
                loss_label = torch.nn.functional.one_hot(
                    label, num_classes=self.num_classes).float()
                out = nn.functional.sigmoid(out)
            elif self.type_criterion == 'binary_mse':
                loss_label = label.float().unsqueeze(1)
            else:
                loss_label = label
            loss = self.criterion(out, loss_label)

        if self.type_criterion == 'binary_mse':
            # decision based on sign (alternatively, check the distance to 1 and -1?)
            out_zeros = torch.zeros_like(out)
            out = torch.concat([out_zeros, out], dim=-1)
            zero_one_labels = out.argmax(-1)  # [B, 2]
            out = torch.where(zero_one_labels == 0, -1, 1)
            acc = torch.eq(out, label).float().mean()
        else:
            acc = torch.eq(out.argmax(-1), label).float().mean()

        self.log("loss", loss)
        self.log("acc", acc)
        if self.report_every > 0 and (batch_idx % self.report_every == 0):
            self.loginf(f'[epoch {self.current_epoch}] Train batch loss: {loss}')
            self.loginf(f'[epoch {self.current_epoch}] Train batch acc {100 * acc} %')
            if self.use_wandb:
                wandb.log({"train_batch_loss": loss,
                           "train_batch_acc": 100 * acc,})
        return loss

    def training_epoch_end(self, outputs):
        self.log("lr", self.optimizer.param_groups[0]["lr"], on_epoch=self.current_epoch)
        self.loginf(f'[epoch {self.current_epoch}] Learning rate: {self.optimizer.param_groups[0]["lr"]}')
        if self.use_wandb:
            wandb.log({"learning_rate": self.optimizer.param_groups[0]["lr"]})

        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for data in valid_dl:
                images, labels = data
                images, labels = images.to('cuda'), labels.to('cuda')
                outputs = self.model(images)

                if self.type_criterion == 'binary_mse':
                    # decision based on sign (alternatively, check the distance to 1 and -1?)
                    out_zeros = torch.zeros_like(outputs)
                    outputs = torch.concat([out_zeros, outputs], dim=-1)
                    zero_one_labels = outputs.argmax(-1)
                    predicted = torch.where(zero_one_labels == 0, -1, 1)
                else:
                    _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            val_acc = 100 * correct / total
        self.loginf(f'[epoch {self.current_epoch}] Validation Accuracy: {val_acc} %')

        if self.use_wandb:
            wandb.log({"val_acc": val_acc})

        if self.best_val_acc < val_acc:
            self.best_val_acc = val_acc
            self.loginf(
                f'[epoch {self.current_epoch}] Current best validation accuracy: {val_acc} %')
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

        # to be removed.
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for data in test_dl:
                images, labels = data
                images, labels = images.to('cuda'), labels.to('cuda')
                outputs = self.model(images)
                if self.type_criterion == 'binary_mse':
                    # decision based on sign (alternatively, check the distance to 1 and -1?)
                    out_zeros = torch.zeros_like(outputs)
                    outputs = torch.concat([out_zeros, outputs], dim=-1)
                    zero_one_labels = outputs.argmax(-1)  # [B, 2]
                    predicted = torch.where(zero_one_labels == 0, -1, 1)
                else:
                    _, predicted = outputs.max(1)
                # _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            val_acc = 100 * correct / total
        self.loginf(f'[epoch {self.current_epoch}] Test Accuracy: {val_acc} %')
        if self.use_wandb:
            wandb.log({"test_acc": val_acc})

        self.model.train()

    def validation_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        if self.type_criterion == 'mse':
            loss_label = torch.nn.functional.one_hot(
                label, num_classes=self.num_classes).float()
        elif self.type_criterion == 'sigm_mse':
            loss_label = torch.nn.functional.one_hot(
                label, num_classes=self.num_classes).float()
            out = nn.functional.sigmoid(out)
        elif self.type_criterion == 'binary_mse':
            loss_label = label.float().unsqueeze(1)
        else:
            loss_label = label
        loss = self.criterion(out, loss_label)
        if self.type_criterion == 'binary_mse':
            # decision based on sign (alternatively, check the distance to 1 and -1?)
            out_zeros = torch.zeros_like(out)
            out = torch.concat([out_zeros, out], dim=-1)
            zero_one_labels = out.argmax(-1)  # [B, 2]
            out = torch.where(zero_one_labels == 0, -1, 1)
            acc = torch.eq(out, label).float().mean()
        else:
            acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        if self.use_wandb:
            wandb.log({"val_batch_loss": loss,
                       "val_batch_acc": 100 * acc,})

        return loss

    def test_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        if self.type_criterion == 'mse':
            loss_label = torch.nn.functional.one_hot(
                label, num_classes=self.num_classes).float()
        elif self.type_criterion == 'sigm_mse':
            loss_label = torch.nn.functional.one_hot(
                label, num_classes=self.num_classes).float()
            out = nn.functional.sigmoid(out)
        elif self.type_criterion == 'binary_mse':
            loss_label = label.float().unsqueeze(1)
        else:
            loss_label = label
        loss = self.criterion(out, loss_label)
        if self.type_criterion == 'binary_mse':
            # decision based on sign (alternatively, check the distance to 1 and -1?)
            out_zeros = torch.zeros_like(out)
            out = torch.concat([out_zeros, out], dim=-1)
            zero_one_labels = out.argmax(-1)  # [B, 2]
            out = torch.where(zero_one_labels == 0, -1, 1)
            acc = torch.eq(out, label).float().mean()
        else:
            acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        if self.use_wandb:
            wandb.log({"test_batch_loss": loss,
                       "test_batch_acc": 100 * acc,})
        return loss

    def test_epoch_end(self, outputs):
        # last checkpoint
        self.loginf('Test, loading last checkpoint')
        assert self.last_checkpoint is not None
        self.model.load_state_dict(self.last_checkpoint)
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for data in test_dl:
                images, labels = data
                images, labels = images.to('cuda'), labels.to('cuda')
                outputs = self.model(images)
                if self.type_criterion == 'binary_mse':
                    # decision based on sign (alternatively, check the distance to 1 and -1?)
                    out_zeros = torch.zeros_like(outputs)
                    outputs = torch.concat([out_zeros, outputs], dim=-1)
                    zero_one_labels = outputs.argmax(-1)  # [B, 2]
                    predicted = torch.where(zero_one_labels == 0, -1, 1)
                else:
                    _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            val_acc = 100 * correct / total
        self.loginf(f'[last model] Test Accuracy: {val_acc} %')

        if args.binary_task_only_two_classes or args.binary_task:
            assert self.type_criterion == 'binary_mse'
            self.loginf('Evaluate training loss...')
            sum_mse_fn = nn.MSELoss(reduction='sum')
            correct = 0
            total_loss = 0
            total = 0
            with torch.no_grad():
                for data in train_dl:
                    images, labels = data
                    images, labels = images.to('cuda'), labels.to('cuda')
                    outputs = self.model(images)
                    loss = sum_mse_fn(outputs, labels.float().unsqueeze(1))
                    total_loss += loss.item()

                    # decision based on sign (alternatively, check the distance to 1 and -1?)
                    out_zeros = torch.zeros_like(outputs)
                    outputs = torch.concat([out_zeros, outputs], dim=-1)
                    zero_one_labels = outputs.argmax(-1)  # [B, 2]
                    predicted = torch.where(zero_one_labels == 0, -1, 1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                acc = 100 * correct / total
                total_loss = total_loss / total
            self.loginf(f'[last model] Training Accuracy: {acc} %')
            self.loginf(f'[last model] Training Loss: {total_loss}')

        # Best 
        self.loginf('Test, loading best checkpoint')
        assert self.best_checkpoint is not None
        self.model.load_state_dict(self.best_checkpoint)
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for data in test_dl:
                images, labels = data
                images, labels = images.to('cuda'), labels.to('cuda')
                outputs = self.model(images)
                if self.type_criterion == 'binary_mse':
                    # decision based on sign (alternatively, check the distance to 1 and -1?)
                    out_zeros = torch.zeros_like(outputs)
                    outputs = torch.concat([out_zeros, outputs], dim=-1)
                    zero_one_labels = outputs.argmax(-1)  # [B, 2]
                    predicted = torch.where(zero_one_labels == 0, -1, 1)
                else:
                    _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            val_acc = 100 * correct / total
        self.loginf(f'[best model] Test Accuracy: {val_acc} %')

        if args.binary_task_only_two_classes or args.binary_task:
            assert self.type_criterion == 'binary_mse'
            self.loginf('Evaluate training loss...')
            sum_mse_fn = nn.MSELoss(reduction='sum')
            correct = 0
            total_loss = 0
            total = 0
            with torch.no_grad():
                for data in train_dl:
                    images, labels = data
                    images, labels = images.to('cuda'), labels.to('cuda')
                    outputs = self.model(images)
                    loss = sum_mse_fn(outputs, labels.float().unsqueeze(1))
                    total_loss += loss.item()

                    # decision based on sign (alternatively, check the distance to 1 and -1?)
                    out_zeros = torch.zeros_like(outputs)
                    outputs = torch.concat([out_zeros, outputs], dim=-1)
                    zero_one_labels = outputs.argmax(-1)  # [B, 2]
                    predicted = torch.where(zero_one_labels == 0, -1, 1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                acc = 100 * correct / total
                total_loss = total_loss / total
            self.loginf(f'[best model] Training Accuracy: {acc} %')
            self.loginf(f'[best model] Training Loss: {total_loss}')

    def _log_image(self, image):
        grid = torchvision.utils.make_grid(image, nrow=4)
        self.logger.experiment.log_image(grid.permute(1,2,0))
        print("[INFO] LOG IMAGE!!!")


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
    config.freeze_value = args.freeze_value
    config.limit_training_points = args.limit_training_points
else:
    use_wandb = False

# logging
log_file_name = f"{args.work_dir}/log.txt"
handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
logging.basicConfig(
    level=logging.INFO, format='%(message)s', handlers=handlers)

loginf = logging.info

loginf(f"Command executed: {sys.argv[:]}")
loginf(f"Args: {json.dumps(args.__dict__, indent=2)}")

loginf(f"torch version: {torch.__version__}")
loginf(f"Work dir: {args.work_dir}")
loginf(f"Seed: {args.seed}")

args.best_model_path = os.path.join(args.work_dir, 'best_model.pt')
args.latest_model_path = os.path.join(args.work_dir, 'latest_model.pt')
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

net = Net(args)

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
