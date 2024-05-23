# Originally forked from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
# and modified.
#
# MIT License for the original code
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import math

from my_model import BaseModel

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


########
# Adapted from:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class SinPositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=65):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, dim)
        self.register_buffer('pe', pe)  # Will not be trained.

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, dim]
            output: [batch size, sequence length, dim]
        """
        assert x.size(1) < self.max_len, (
            f"Too long sequence length: increase `max_len` of pos encoding")
        # shape of x (len, B, dim)
        return x + self.pe[:, :x.size(1)]


def get_sin_pos(d_model, max_len=65, normalize=True, eps=1e-6):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        mean, std = pe.mean(), pe.std()
        pe = (pe - mean) / (std + eps)
    pe = pe.unsqueeze(0)  # shape (1, max_len, dim)
    return pe


class DualHeadNoLearningInputConstrainedLinearAttention(nn.Module):
    def __init__(self, d_model, d_model_zero,
                 num_heads=12, num_sum_heads=1, dim_head=32,
                 qk_dim_head=32, use_softmax=True, add_diagonal=True,
                 remove_diag_scale=False, learn_attention=False,
                 freeze_value=False, dropout=0.0):
        super().__init__()

        self.d_model = d_model  # N
        self.d_model_zero = d_model_zero  # N_0 for q/k input dim
        self.num_heads = num_heads  # H
        self.dim_head = dim_head  # M
        # assert d_model // num_heads == dim_head  # M = N / H
        self.qk_dim_head = qk_dim_head  # G
        self.num_sum_heads = num_sum_heads  # F
        self.num_heads_eff = num_heads * num_sum_heads  # H * F

        # scalers
        self.attn_scale = qk_dim_head ** -0.5
        self.sum_head_scale = num_sum_heads ** -0.5

        self.use_softmax = use_softmax
        if use_softmax:
            self.softmax = nn.Softmax(dim = -1)

        self.to_v = nn.Linear(d_model, dim_head * self.num_heads_eff, bias=False)
        self.to_q = nn.Linear(
            d_model_zero, qk_dim_head * self.num_heads_eff, bias=False)
        self.to_k = nn.Linear(
            d_model_zero, qk_dim_head * self.num_heads_eff, bias=False)
        
        self.v_dropout = nn.Dropout(dropout, inplace=True)
        self.q_dropout = nn.Dropout(dropout, inplace=True)
        self.k_dropout = nn.Dropout(dropout, inplace=True)

        # Freeze Q, K
        if not learn_attention:
            for param in self.to_q.parameters():
                param.requires_grad = False
            for param in self.to_k.parameters():
                param.requires_grad = False
            self.init_qk(add_diagonal, remove_diag_scale)

        if freeze_value:
            for param in self.to_v.parameters():
                param.requires_grad = False

        # out linear projection in all cases except when H=1 and M=N
        if num_heads == 1 and dim_head == d_model:
            self.to_out = nn.Identity()
        else:
            self.to_out = nn.Linear(dim_head * num_heads, d_model, bias=False)

    def init_qk(self, add_diagonal=True, remove_diag_scale=False):
        num_rows, num_cols = self.to_q.weight.shape
        assert num_cols == self.d_model_zero
        assert num_rows == self.qk_dim_head * self.num_heads_eff
        std = self.d_model_zero ** -0.5
        id_scaler = 1.0 if remove_diag_scale else self.qk_dim_head ** -0.25 

        # init
        nn.init.normal_(self.to_q.weight, mean=0., std=std)
        nn.init.normal_(self.to_k.weight, mean=0., std=std)
        if add_diagonal:
            if self.num_heads == 1:
                assert self.d_model == self.qk_dim_head, 'necessary for current identity init.'

            assert self.d_model_zero == self.d_model, 'Otherwise diagonal init can not be used.'
            # identity has to be repeated for each sum_head
            repeated_identity = torch.eye(self.d_model).unsqueeze(-1).repeat(self.num_sum_heads, 1, 1) * id_scaler
            repeated_identity = repeated_identity.view(self.d_model * self.num_sum_heads, self.d_model)
            self.to_q.weight += repeated_identity
            self.to_k.weight += repeated_identity

    def forward(self, x, x_0, drop_heads=None):
        # drop_heads is a list of integers

        v = self.v_dropout(self.to_v(x))
        q = self.q_dropout(self.to_q(x_0))
        k = self.k_dropout(self.to_k(x_0))

        k = rearrange(k, 'b t (h g) -> b h t g', h = self.num_heads_eff)
        q = rearrange(q, 'b t (h g) -> b h t g', h = self.num_heads_eff)
        v = rearrange(v, 'b t (h m) -> b h t m', h = self.num_heads_eff)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.attn_scale

        if self.use_softmax:
            attn = self.softmax(attn)

        out = torch.matmul(attn, v)
        # separate summation heads from concatenation heads, and sum over the former:
        out = rearrange(out, 'b (h f) t m -> b h f t m', h = self.num_heads)
        if drop_heads is None:
            out = self.sum_head_scale * out.sum(2)
        else:
            for drop_head_index in drop_heads:
                out[:, :, drop_head_index, :, :] = 0
            sum_head_scale = (self.num_sum_heads - len(drop_heads)) ** -0.5
            out = sum_head_scale * out.sum(2)
        out = rearrange(out, 'b h t m -> b t (h m)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.,
                 no_feedforward=False):
        super().__init__()
        self.no_feedforward = no_feedforward
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            if no_feedforward:
                self.layers.append(
                    Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))
            else:
                self.layers.append(nn.ModuleList([
                    Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(dim, mlp_dim, dropout = dropout)
                ]))

    def forward(self, x):
        if self.no_feedforward:
            for attn in self.layers:
                x = attn(x) + x
        else:
            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x
        return self.norm(x)


class DualHeadNoLearningInputConstrainedLinearTransformer(nn.Module):
    def __init__(self, d_model, d_model_zero, num_layers, num_heads,
                 num_sum_heads, dim_head, qk_dim_head, no_residual=False,
                 remove_diag_scale=False, learn_attention=False,
                 remove_diagonal_init=False, freeze_value=False,
                 dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.no_residual = no_residual
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                DualHeadNoLearningInputConstrainedLinearAttention(
                    d_model=d_model, d_model_zero=d_model_zero,
                    num_heads=num_heads,
                    num_sum_heads=num_sum_heads, dim_head=dim_head,
                    qk_dim_head=qk_dim_head,
                    remove_diag_scale=remove_diag_scale,
                    learn_attention=learn_attention,
                    add_diagonal=(not remove_diagonal_init),
                    freeze_value=freeze_value,
                    dropout=dropout,
                ))

    def forward(self, x, x_0, drop_heads=None):
        if drop_heads is not None:
            drop_heads_per_layer_list = []
            for _ in range(self.num_layers):
                drop_heads_per_layer_list.append(None)
            # format "0-1,0-2"
            for head in drop_heads.split(','):
                layer_id, head_id = head.split('-')
                layer_id = int(layer_id)
                head_id = int(head_id)
                if drop_heads_per_layer_list[layer_id] is None:
                    drop_heads_per_layer_list[layer_id] = [head_id]
                else:
                    drop_heads_per_layer_list[layer_id].append(head_id)

            layer_id = 0
            for attn in self.layers:
                if self.no_residual:
                    x = attn(
                        x, x_0,
                        drop_heads=drop_heads_per_layer_list[layer_id])
                else:
                    x = attn(
                        x, x_0,
                        drop_heads=drop_heads_per_layer_list[layer_id]) + x
                layer_id += 1
        else:
            for attn in self.layers:
                if self.no_residual:
                    x = attn(x, x_0, drop_heads=drop_heads)
                else:
                    x = attn(x, x_0, drop_heads=drop_heads) + x
        return x


class MyViT(BaseModel):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls',
                 channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,
                 use_random_first_projection=False, use_sin_pos_enc=False,
                 in_context_learning=False, freeze_icl_label_embedding=False,
                 icl_num_shots=1, no_feedforward=False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.no_cls = True if pool == 'mean' else False
        extra_pos = 0 if self.no_cls else 1

        if in_context_learning:
            # rearrange, and transpose first two
            self.rearrange_input = Rearrange(
                's b c (h p1) (w p2) -> b s (h w) (p1 p2 c)',
                p1 = patch_height, p2 = patch_width)
            # +1 for cls token is added later in pos_embedding
            num_patches = num_patches * (2 * icl_num_shots + 1)
            self.label_embedding = nn.Embedding(3, dim)
            if freeze_icl_label_embedding:
                for param in self.label_embedding.parameters():
                    param.requires_grad = False
            self.cls_token_label_emb = nn.Parameter(torch.randn(1, 1, dim))
        else:
            self.rearrange_input = Rearrange(
                'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                p1 = patch_height, p2 = patch_width)

        self.to_patch_embedding_in_ln = nn.LayerNorm(patch_dim)
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.to_patch_embedding_out_ln = nn.LayerNorm(dim)

        if use_random_first_projection:
            for param in self.to_patch_embedding.parameters():
                param.requires_grad = False

        self.in_context_learning = in_context_learning
        self.icl_num_shots = icl_num_shots

        if use_sin_pos_enc:
            self.pos_embedding = nn.Parameter(
                get_sin_pos(dim, num_patches + extra_pos, normalize=True),
                requires_grad=False)
        else:
            self.pos_embedding = nn.Parameter(
                torch.randn(1, num_patches + extra_pos, dim))

        if not self.no_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout,
            no_feedforward=no_feedforward)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img, icl_labels=None):
        x = self.rearrange_input(img)
        x = self.to_patch_embedding_in_ln(x)
        x = self.to_patch_embedding(x)
        x = self.to_patch_embedding_out_ln(x)
        if self.in_context_learning:
            bsz, num_icl_examples, num_patches, _ = x.shape
            n = num_patches * num_icl_examples
            # merge number of shots and number of tokens:
            x = rearrange(x, 'b s t d -> b (s t) d')
        else:
            bsz, num_patches, _ = x.shape
            n = num_patches

        if not self.no_cls:
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = bsz)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
        else:
            x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        if self.in_context_learning:
            assert icl_labels is not None
            icl_labels = repeat(
                icl_labels.unsqueeze(-1), 's b 1 -> s b t', t = num_patches)
            icl_labels = rearrange(icl_labels, 's b t -> b (s t)')
            icl_labels = self.label_embedding(icl_labels)
            if not self.no_cls:
                cls_emb = repeat(self.cls_token_label_emb, '1 1 d -> b 1 d', b = bsz)
                icl_labels = torch.cat((cls_emb, icl_labels), dim=1)
            x += icl_labels

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


# our model
class DualHeadNoLearningInputConstrainedLinearViT(BaseModel):
    def __init__(self, *, image_size, patch_size, num_classes,
                 d_model, num_layers, num_heads, num_sum_heads,
                 pool='cls', channels=3, dim_head=64, qk_dim_head=64,
                 concat_pos_emb_dim=96, use_random_first_projection=False,
                 use_sin_pos_enc=False, no_cls_token=False, no_residual=False,
                 concat_pos_enc=False, use_random_position_encoding=False,
                 remove_diag_scale=False, learn_attention=False,
                 use_parallel=False, add_learned_input_layer=False,
                 remove_diagonal_init=False,
                 remove_nonlinear_input_projection=False,
                 in_context_learning=False,
                 additive_icl_label_embedding=False,
                 freeze_icl_label_embedding=False,
                 icl_num_shots=1, readout_column_index=0,
                 freeze_value=False, freeze_input_projection=False,
                 freeze_readout=False, add_biases=False):
        super().__init__()

        self.readout_column_index = readout_column_index
        self.icl_num_shots = icl_num_shots

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        # set sequence length
        self.no_cls_token = no_cls_token
        if no_cls_token:
            if in_context_learning:
                # num_shots * number of classes (2 for binary) examples plus one unknown
                max_seq_len = num_patches * (2 * icl_num_shots + 1)
            else:
                max_seq_len = num_patches
        else:
            max_seq_len = num_patches + 1

        # input is augmented with one hot label repr.
        x_zero_extra_dim = 3 if in_context_learning else 0

        # positional encoding
        self.concat_pos_enc = concat_pos_enc
        # set pos embedding dimension
        if concat_pos_enc:
            # concatenation
            self.pos_emb_dim = concat_pos_emb_dim
            patch_proj_dim = d_model - concat_pos_emb_dim
            assert not remove_nonlinear_input_projection
        else:
            # sum
            if remove_nonlinear_input_projection:
                self.pos_emb_dim = patch_dim
                patch_proj_dim = patch_dim
            else:
                self.pos_emb_dim = d_model
                patch_proj_dim = d_model

        if use_sin_pos_enc:
            self.pos_embedding = nn.Parameter(
                get_sin_pos(self.pos_emb_dim, max_seq_len, normalize=True),
                requires_grad=False)

        elif use_random_position_encoding:
            self.pos_embedding = nn.Parameter(
                torch.randn(1, max_seq_len, self.pos_emb_dim), requires_grad=False)
        else:
            self.pos_embedding = nn.Parameter(
                torch.randn(1, max_seq_len, self.pos_emb_dim))

        if not no_cls_token:
            # learn cls embedding
            self.cls_token = nn.Parameter(torch.randn(1, 1, patch_proj_dim))

        self.in_context_learning = in_context_learning
        self.additive_icl_label_embedding = additive_icl_label_embedding
        if additive_icl_label_embedding:
            assert in_context_learning
            x_zero_extra_dim = 0
            self.label_embedding = nn.Embedding(3, self.pos_emb_dim)
            # self.label_embedding = nn.Linear(3, self.pos_emb_dim)
            if freeze_icl_label_embedding:
                for param in self.label_embedding.parameters():
                    param.requires_grad = False

        if in_context_learning:
            # rearrange, and transpose first two
            self.rearrange_input = Rearrange(
                's b c (h p1) (w p2) -> b s (h w) (p1 p2 c)',
                p1 = patch_height, p2 = patch_width)
        else:
            self.rearrange_input = Rearrange(
                'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                p1 = patch_height, p2 = patch_width)

        if remove_nonlinear_input_projection:
            self.to_patch_embedding = nn.Sequential(
                nn.LayerNorm(patch_dim),
            )
            d_model_zero = patch_dim + x_zero_extra_dim
        else:
            self.to_patch_embedding = nn.Sequential(
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, patch_proj_dim),
                nn.Tanh(),
            )
            d_model_zero = d_model + x_zero_extra_dim

        # The name is now redundant with `remove_random_nonlinear_projection`
        # but keeping it as is for now; maybe rename to "freeze_input_proj"
        if use_random_first_projection:
            for param in self.to_patch_embedding.parameters():
                param.requires_grad = False

        self.add_learned_input_layer = add_learned_input_layer
        # this is after the positional encoding.
        # NB: with this layer, we can have a separate "N_0" (not tied to d_model)
        # for now it is kept this way.
        if add_learned_input_layer:
            self.learned_input_layer = nn.Linear(
                d_model_zero, d_model, bias=add_biases)
            if freeze_input_projection:
                for param in self.learned_input_layer.parameters():
                    param.requires_grad = False

        if use_parallel:
            assert d_model_zero == d_model, 'not implemented yet'
            assert freeze_value is True, 'not implemented yet'
            assert False
        else:
            self.transformer = DualHeadNoLearningInputConstrainedLinearTransformer(
                d_model, d_model_zero, num_layers, num_heads, num_sum_heads,
                dim_head, qk_dim_head, no_residual, remove_diag_scale,
                learn_attention=learn_attention,
                remove_diagonal_init=remove_diagonal_init,
                freeze_value=freeze_value)

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool

        self.readout_layer = nn.Linear(d_model, num_classes, bias=add_biases)
        if freeze_readout:
            for param in self.readout_layer.parameters():
                param.requires_grad = False  

    def forward(self, img, get_x_zero=False, icl_labels=None, drop_heads=None):
        x = self.rearrange_input(img)
        x = self.to_patch_embedding(x)

        if self.in_context_learning:
            bsz, num_icl_examples, num_patches, _ = x.shape
        else:
            bsz, num_patches, _ = x.shape

        if self.no_cls_token:
            if self.in_context_learning:
                slen = num_patches * num_icl_examples
            else:
                slen = num_patches
        else:
            assert not self.in_context_learning, 'Not supported yet.'
            slen = num_patches + 1
            cls_tokens = repeat(self.cls_token, '1 1 n -> b 1 n', b = bsz)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.in_context_learning:
            # merge number of shots and number of tokens:
            x = rearrange(x, 'b s t d -> b (s t) d')

        if self.concat_pos_enc:
            pos_emb = repeat(
                self.pos_embedding[:, :slen], '1 t k -> b t k', b = bsz)
            x = torch.cat([x, pos_emb], dim=-1)
        else:
            x += self.pos_embedding[:, :slen]  # TODO incontext

        if self.in_context_learning:
            assert icl_labels is not None
            # now append the labels
            if self.additive_icl_label_embedding:
                icl_labels = repeat(
                    icl_labels.unsqueeze(-1), 's b 1 -> s b t', t = num_patches)
                icl_labels = rearrange(icl_labels, 's b t -> b (s t)')
                x += self.label_embedding(icl_labels)
            else:
                icl_labels = torch.nn.functional.one_hot(
                    icl_labels, num_classes=3)  # binary + unk token (B, S, 3)
                # duplicate label for all tokens belonging to the same image
                icl_labels = repeat(
                    icl_labels.unsqueeze(-2), 's b 1 d -> s b t d', t = num_patches)
                icl_labels = rearrange(icl_labels, 's b t d -> b (s t) d')
                x = torch.cat([x, icl_labels], dim=-1)

        # copy the x0 layer input
        x_0 = x.clone()

        if self.add_learned_input_layer:
            x = self.learned_input_layer(x)

        x = self.transformer(x, x_0, drop_heads=drop_heads)  # (b, t, m=n)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, self.readout_column_index]

        if get_x_zero:
            return (self.readout_layer(x), x_0)
        else:
            return self.readout_layer(x)


class DualHeadNoLearningInputConstrainedLinearTextTrafo(BaseModel):
    def __init__(self, *, input_vocab_size, token_embedding_dim, max_seq_len, num_classes,
                 d_model, num_layers, num_heads, num_sum_heads,
                 pool='cls', dim_head=64, qk_dim_head=64,
                 concat_pos_emb_dim=96,
                 use_sin_pos_enc=False, no_cls_token=False, no_residual=False,
                 concat_pos_enc=False, use_random_position_encoding=False,
                 remove_diag_scale=False, learn_attention=False,
                 add_learned_input_layer=False,
                 remove_diagonal_init=False,
                 remove_nonlinear_input_projection=False,
                 readout_column_index=0,
                 freeze_value=False, freeze_input_projection=False,
                 freeze_readout=False, add_biases=False, dropout=0.0):
        super().__init__()

        self.readout_column_index = readout_column_index

        patch_dim = token_embedding_dim

        # token embedding
        self.token_embedding = nn.Embedding(input_vocab_size, patch_dim)
        self.emb_dropout = nn.Dropout(dropout, inplace=True)

        # positional encoding
        self.concat_pos_enc = concat_pos_enc
        # set pos embedding dimension
        if concat_pos_enc:
            # concatenation
            self.pos_emb_dim = concat_pos_emb_dim
            patch_proj_dim = d_model - concat_pos_emb_dim
            assert not remove_nonlinear_input_projection
        else:
            # sum
            if remove_nonlinear_input_projection:
                self.pos_emb_dim = patch_dim
                patch_proj_dim = patch_dim
            else:
                self.pos_emb_dim = d_model
                patch_proj_dim = d_model

        if use_sin_pos_enc:
            self.pos_embedding = nn.Parameter(
                get_sin_pos(self.pos_emb_dim, max_seq_len, normalize=True),
                requires_grad=False)
        elif use_random_position_encoding:
            self.pos_embedding = nn.Parameter(
                torch.randn(1, max_seq_len, self.pos_emb_dim), requires_grad=False)
        else:
            self.pos_embedding = nn.Parameter(
                torch.randn(max_seq_len, self.pos_emb_dim))

        self.no_cls_token = no_cls_token
        if not no_cls_token:
            # learn cls embedding
            self.cls_token = nn.Parameter(torch.randn(1, 1, patch_proj_dim))

        d_model_zero = patch_dim

        self.add_learned_input_layer = add_learned_input_layer
        if add_learned_input_layer:
            self.learned_input_layer = nn.Linear(
                d_model_zero, d_model, bias=add_biases)
            if freeze_input_projection:
                for param in self.learned_input_layer.parameters():
                    param.requires_grad = False
            self.input_dropout = nn.Dropout(dropout, inplace=True)

        self.transformer = DualHeadNoLearningInputConstrainedLinearTransformer(
            d_model, d_model_zero, num_layers, num_heads, num_sum_heads,
            dim_head, qk_dim_head, no_residual, remove_diag_scale,
            learn_attention=learn_attention,
            remove_diagonal_init=remove_diagonal_init,
            freeze_value=freeze_value, dropout=dropout)

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool

        self.readout_layer = nn.Linear(d_model, num_classes, bias=add_biases)
        if freeze_readout:
            for param in self.readout_layer.parameters():
                param.requires_grad = False  

    def forward(self, img, get_x_zero=False, drop_heads=None):
        x = self.token_embedding(img).transpose(0, 1)
        x = self.emb_dropout(x)

        bsz, num_patches, _ = x.shape

        if self.no_cls_token:
            slen = num_patches
        else:
            assert False, 'Not supported yet.'

        if self.concat_pos_enc:
            pos_emb = repeat(
                self.pos_embedding[:, :slen], '1 t k -> b t k', b = bsz)
            x = torch.cat([x, pos_emb], dim=-1)
        else:
            x += self.pos_embedding.unsqueeze(0)[:, :slen]

        # copy the x0 layer input
        x_0 = x.clone()

        if self.add_learned_input_layer:
            x = self.learned_input_layer(x)
            x = self.input_dropout(x)

        x = self.transformer(x, x_0, drop_heads=drop_heads)  # (b, t, m=n)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, self.readout_column_index]

        if get_x_zero:
            return (self.readout_layer(x), x_0)
        else:
            return self.readout_layer(x)
