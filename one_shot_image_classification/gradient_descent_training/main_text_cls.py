import argparse
import random
import sys
import os
import json
import logging
import hashlib
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
import numpy as np
from einops import repeat

import warmup_scheduler

import copy

from utils import get_model, get_criterion
from utils_text_data import ImdbDataset

from utils import get_experiment_name

parser = argparse.ArgumentParser()
parser.add_argument("--show_progress_bar", action="store_true")
parser.add_argument("--dataset", default="omniglot", type=str,
                    help="[omniglot, miniimagenet]")
parser.add_argument("--load_embedding_from", default=None, type=str)
parser.add_argument("--no_load_pos", action="store_true")
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
parser.add_argument("--data_device", default="cuda", type=str)

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
    max_seq_length=args.max_seq_length, pad_idx=args.pad_idx,
    device=args.data_device)

valid_data = ImdbDataset(
    pos_data_file=valid_pos_file, neg_data_file=valid_neg_file,
    max_seq_length=args.max_seq_length, pad_idx=args.pad_idx,
    device=args.data_device)

# Set dataloader
train_dl = DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(
    dataset=valid_data, batch_size=eval_batch_size, shuffle=False)

# test set
test_pos_file = f"{data_path}/test.pos.bpe.txt"
test_neg_file = f"{data_path}/test.neg.bpe.txt"
test_data = ImdbDataset(
    pos_data_file=test_pos_file, neg_data_file=test_neg_file,
    max_seq_length=args.max_seq_length, pad_idx=args.pad_idx,
    device=args.data_device)
test_dl = DataLoader(
    dataset=test_data, batch_size=eval_batch_size, shuffle=False)

# create a new dataloader with limited datapoints.
if args.limit_training_points:
    assert False, 'not implemented yet'


class TextClassificationNet(pl.LightningModule):
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

        if args.load_embedding_from is not None:
            loginf(f"Init token and position embeddings from: "
                   f"{args.load_embedding_from}")
            checkpoint = torch.load(
                args.load_embedding_from, map_location={'cuda:0': 'cpu'})
            checkpoint_dict = checkpoint['model']
            model_dict = self.model.state_dict()
            new_dict = {}
            # TODO change the map name
            for key, value in model_dict.items():  # 2-dim
                if 'token_embedding' in key:
                    loginf(f"loading: {key}")
                    new_dict[key] = checkpoint_dict['decoder.sentence_encoder.embed_tokens.weight']
                elif 'pos_embedding' in key and not args.no_load_pos:
                    loginf(f"loading: {key}")
                    new_dict[key] = checkpoint_dict['decoder.sentence_encoder.embed_positions.weight'][:args.max_seq_length]
            model_dict.update(new_dict)
            self.model.load_state_dict(model_dict)

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
        # img, label = batch
        # construct net input
        inputs, target_labels = batch

        inputs = inputs.to('cuda')  # (B, len, 1, 28, 28)
        target_labels = target_labels.to('cuda')  # (B, len)
        # bsz, slen = input_labels.shape

        inputs = inputs.transpose(0, 1)

        out = self.model(inputs)

        # convert labels to +1/-1
        target_labels = 2 * target_labels - 1.

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

    # def training_epoch_end(self, outputs):
    #     self.log("lr", self.optimizer.param_groups[0]["lr"], on_epoch=self.current_epoch)

    def training_epoch_end(self, outputs=None, step=None):
        self.log("lr", self.optimizer.param_groups[0]["lr"], on_epoch=self.current_epoch)
        self.loginf(f'[epoch {self.current_epoch}] Learning rate: {self.optimizer.param_groups[0]["lr"]}')
        if self.use_wandb:
            wandb.log({"learning_rate": self.optimizer.param_groups[0]["lr"]})

        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for data in valid_dl:
                inputs, target_labels = data
                inputs = inputs.to('cuda')  # (B, len, 1, 28, 28)
                target_labels = target_labels.to('cuda')  # (B, len)
                # bsz, slen = input_labels.shape

                inputs = inputs.transpose(0, 1)

                # convert labels to +1/-1
                target_labels = 2 * target_labels - 1.

                outputs = self.model(inputs)

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
                # iter += 1
                # if iter > self.icl_eval_num_iter:
                #     break
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
            # print(f'saving the best checkpoint {self.best_checkpoint}')
            self.best_checkpoint = copy.deepcopy(self.model).state_dict()

        torch.save({'epoch': self.current_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'valid_acc': val_acc}, self.latest_model_path)
        self.last_checkpoint = copy.deepcopy(self.model).state_dict()

        # test for debugging purpose
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_dl:
                inputs, target_labels = data
                inputs = inputs.to('cuda')  # (B, len, 1, 28, 28)
                target_labels = target_labels.to('cuda')  # (B, len)
                # bsz, slen = input_labels.shape

                inputs = inputs.transpose(0, 1)

                # convert labels to +1/-1
                target_labels = 2 * target_labels - 1.

                outputs = self.model(inputs)

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
                # iter += 1
                # if iter > self.icl_eval_num_iter:
                #     break
            val_acc = 100 * correct / total
        self.loginf(f'[epoch {self.current_epoch}, step {step}] Test Accuracy: {val_acc} %')

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

                inputs, target_labels = data
                inputs = inputs.to('cuda')  # (B, len, 1, 28, 28)
                target_labels = target_labels.to('cuda')  # (B, len)

                inputs = inputs.transpose(0, 1)

                # convert labels to +1/-1
                target_labels = 2 * target_labels - 1.

                outputs = self.model(inputs)

                # images, labels = data
                # images, labels = images.to('cuda'), labels.to('cuda')
                # outputs = self.model(images)
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
                # if iter > self.icl_eval_num_iter:
                #     break
            val_acc = 100 * correct / total
        self.loginf(f'[last model] Test Accuracy: {val_acc} %')

        assert self.type_criterion == 'binary_mse'
        self.loginf('Evaluate training loss...')
        sum_mse_fn = nn.MSELoss(reduction='sum')
        correct = 0
        total_loss = 0
        total = 0
        iter = 0
        with torch.no_grad():
            for data in train_dl:
                inputs, target_labels = data

                # images, labels = data
                # images, labels = images.to('cuda'), labels.to('cuda')
                # inputs, input_labels = data['train']
                inputs = inputs.to('cuda')  # (B, len, 1, 28, 28)
                target_labels = target_labels.to('cuda')  # (B, len)
                # bsz, slen = input_labels.shape

                inputs = inputs.transpose(0, 1)

                # convert labels to +1/-1
                target_labels = 2 * target_labels - 1.

                outputs = self.model(inputs)

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
                # if iter > self.icl_eval_num_iter:
                #     break
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
        iter = 0
        self.model.eval()
        with torch.no_grad():
            for data in test_dl:
                inputs, target_labels = data
                inputs = inputs.to('cuda')  # (B, len, 1, 28, 28)
                target_labels = target_labels.to('cuda')  # (B, len)

                inputs = inputs.transpose(0, 1)
                target_labels = 2 * target_labels - 1.
                outputs = self.model(inputs)

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
                # if iter > self.icl_eval_num_iter:
                #     break
            val_acc = 100 * correct / total
        self.loginf(f'[best model] Test Accuracy: {val_acc} %')

        assert self.type_criterion == 'binary_mse'
        self.loginf('Evaluate training loss...')
        sum_mse_fn = nn.MSELoss(reduction='sum')
        correct = 0
        total_loss = 0
        total = 0
        iter = 0
        with torch.no_grad():
            for data in train_dl:
                inputs, target_labels = data

                # inputs, input_labels = data['train']
                inputs = inputs.to('cuda')  # (B, len, 1, 28, 28)
                target_labels = target_labels.to('cuda')  # (B, len)
                # bsz, slen = input_labels.shape

                inputs = inputs.transpose(0, 1)

                # convert labels to +1/-1
                target_labels = 2 * target_labels - 1.

                outputs = self.model(inputs)

                loss = sum_mse_fn(outputs, target_labels.float().unsqueeze(1))
                total_loss += loss.item()

                # decision based on sign (alternatively, check the distance to 1 and -1?)
                out_zeros = torch.zeros_like(outputs)
                outputs = torch.concat([out_zeros, outputs], dim=-1)
                zero_one_labels = outputs.argmax(-1)  # [B, 2]
                predicted = torch.where(zero_one_labels == 0, -1, 1)

                total += target_labels.size(0)
                correct += (predicted == target_labels).sum().item()
                # if iter > self.icl_eval_num_iter:
                #     break
            acc = 100 * correct / total
            total_loss = total_loss / total
        self.loginf(f'[best model] Training Accuracy: {acc} %')
        self.loginf(f'[best model] Training Loss: {total_loss}')
        self.loginf('======== END ====================================== ')


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

net = TextClassificationNet(args)

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
