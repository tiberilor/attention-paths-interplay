import torch
import sys
import time
from torchvision import datasets
from torchvision.transforms import ToTensor
import random
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from datetime import datetime
from einops import rearrange, reduce, repeat, einsum
from itertools import product

# UTILITIES FUNCTIONS


def seed_everything(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unseed_everything():
    # set a random seed using the current time
    seed = int(datetime.now().timestamp())
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ATTENTION WEIGHTS GENERATION FUNCTIONS

# THIS IS FOR OPTION A
# TODO: change the name of this
def generate_attention_weights_markov(partial_width, shift=-1.0, pos_encoding_period=1.0e4, version="same_token"):
    # version options: "same_token", "different_token", "random_features", "random_positions", "all_random",
    # "random_features_only"

    # NOTE: the returned w_weights should take key on the left and query on the right

    # generate the feature-informative block:
    if version == "same_token" or version == "random_positions":
        info_block = torch.eye(2*partial_width)/partial_width
    elif version == "different_token":
        id = torch.eye(partial_width)
        zeros = torch.zeros(partial_width, partial_width)
        lower_rettangle = torch.cat([id, zeros], dim=1)
        upper_rettangle = torch.cat([zeros, id], dim=1)
        info_block = torch.cat([upper_rettangle, lower_rettangle], dim=0)/partial_width
    elif version == "random_features" or version == "all_random" or version == "random_features_only":
        info_block = torch.zeros(2*partial_width, 2*partial_width)/partial_width
    else:
        print("WARNING: the specified head version" + version + " is not valid. Exiting")
        sys.exit()

    # generate the noise-features block:
    if version == "random_features" or version == "all_random" or version == "random_features_only":
        noise_block = torch.randn(partial_width, partial_width)/partial_width
    else:
        noise_block = torch.zeros(partial_width, partial_width)/partial_width

    # generate the bos block:
    if version == "random_positions" or version == "all_random" or version == "random_features_only":
        bos_block = torch.zeros(2, 2)
    else:
        frequency = 2*np.pi/pos_encoding_period
        factor = 1.0 + (1.0 + np.cos(frequency))/2
        bos_block = factor * torch.tensor([[0, 1], [0, 0]])

    # generate the position block
    if version == "random_positions" or version == "all_random":
        position_block = torch.randn(2, 2)/2
    elif version == "random_features_only":
        position_block = torch.zeros(2, 2)
    else:
        # this shifts the positional encoding by -1
        position_block = mini_position_shift_matrix(shift=shift, pos_encoding_period=pos_encoding_period)

    # compose the block diagonal attention weights
    w_weights = torch.block_diag(info_block, noise_block, bos_block, position_block)
    # shape is [input_width, input_width], with input_width = 3*partial_width + 4

    # add a dummy head dimension
    w_weights = rearrange(w_weights, "i j -> 1 i j")
    # shape is [number_heads=1, input_width, input_width]

    return w_weights


# THIS IS FOR OPTION B
def generate_attention_weights_markov_optionB(partial_width, max_number_tokens, shift=-1, version="same_token",
                                              informative_perturbation=0.0, noninformative_perturbation=0.0,
                                              info_and_noninfo_perturbation=0.0, positions_perturbation=0.0):
    """
    Returns an attention weight for the Markov chain task

    Parameters
    ----------
    partial_width: int
        Half the width of the informative features part of the input.
        For reference, the input is of size [informative_width + noninformative_width + one-hot_positions_width],
        with sizes:
            -> informative_width = 2*partial_width;
            -> noninformative_width = = 2*partial_width;
            -> one-hot_positions_width = max_number_tokens + 1 (the +1 is for the bos token)

    max_number_tokens: int
        Max sequence length allowed by the model.
        This defines the length of the one-hot positional encoding part of the input (see partial_width definition)

    shift: int
        Only relevant for the good heads "same_token" and "different_token". Defines the relative distance of the key
        token to which the query token attends to.
        The good heads have a shift of -/+ 1.

    version: str
        Can be one of the following:
            -> "same_token": good head, checks if the token at distance <shift> is the same as query, otherwise attends
            the bos token (i.e. nothing: bos token is zeros in the features space)
            -> "different_token": analogous to "same_token", but check is the key token is different than query
            -> "uniform attention": Attends all tokens equally. This is done by ignores the features space, and applying
            a uniform matrix of ones on the positions space.
            -> "blank": attention weights are zero. This can be used to generate random attention weights,
            by controlling the perturbation strength parameters.

    informative_perturbation: float
        adds a perturbation to the block of attention weights acting on the informative features subspace

    noninformative_perturbation: float
        adds a perturbation to the block of attention weights acting on the noninformative features subspace

    info_and_noninfo_perturbation: float
        adds a perturbation to the attention weights acting on the whole features subspace.
        The difference with perturbing just the informative and noninformative blocks separately, is that here the
        attention weights will also mix the informative and noninformative features subspace.
        NOTE: the positions subspace is never mixed with the others, instead. It is of a different nature (one-hot),
        than the featrures subsapce, so it should be treated separately.

    positions_perturbation: float
        adds a perturbation to the attention weights acting on the positional encoding subspace.
    """

    # version options: "same_token", "different_token", "uniform_attention", "blank".

    # Lorenzo: Here we define the parameters establishing the strength of the info and position part of the good head.
    # These are set to 1 without loss of generality, but I leave them here as a tunable parameter for completeness and
    # to match my notes on the derivation of the good head.
    informative_features_strength = 1.0
    position_strength = 1.0

    # NOTE: the returned w_weights should take key on the left and query on the right

    # generate the feature-informative block:
    if version == "same_token":
        info_block = torch.eye(2*partial_width)/partial_width
        # multiply by strength
        info_block *= informative_features_strength
    elif version == "different_token":
        id = torch.eye(partial_width)
        zeros = torch.zeros(partial_width, partial_width)
        lower_rectangle = torch.cat([id, zeros], dim=1)
        upper_rectangle = torch.cat([zeros, id], dim=1)
        info_block = torch.cat([upper_rectangle, lower_rectangle], dim=0)/partial_width
        # multiply by strength
        info_block *= informative_features_strength
    elif version == "blank" or version == "uniform_attention":
        info_block = torch.zeros(2*partial_width, 2*partial_width)/partial_width
    else:
        print("WARNING: the specified head version" + version + " is not valid. Exiting")
        sys.exit()
    # add perturbation
    info_block += informative_perturbation * torch.randn_like(info_block)/partial_width

    # generate the noise-features block:
    # same for all: "same_token", "different_token", "uniform_attention", "blank"
    if noninformative_perturbation > 0:
        noise_block = noninformative_perturbation * torch.randn(partial_width, partial_width)/partial_width
    else:
        noise_block = torch.zeros(partial_width, partial_width) / partial_width

    # generate the position block:
    if version == "blank":
        position_block = torch.zeros(max_number_tokens + 1, max_number_tokens + 1)
    elif version == "uniform_attention":
        position_block = torch.ones(max_number_tokens + 1, max_number_tokens + 1)
    else:  # "same_token", "different_token"
        # generate the position shift block
        offset = int(-1*shift)
        position_block = torch.diag_embed(torch.diagonal(torch.ones(max_number_tokens, max_number_tokens), offset=offset),
                                          offset=offset)
        # size [max_number_tokens, max_number_tokens]
        # multiply by strength
        position_block *= position_strength
        # add row-column for the bos position
        position_block = torch.block_diag(torch.zeros(1, 1), position_block)
        # size [max_number_tokens + 1, max_number_tokens + 1]
        # add attention to bos token from any other token (i.e. row [0, 1, 1, 1, 1, ..., 1])
        bos_strength = (position_strength + informative_features_strength +
                        max(position_strength, informative_features_strength))/2
        bos_row = torch.ones(max_number_tokens + 1) * bos_strength
        bos_row[0] = 0.0  # we don't want the bos token to attend to itself
        position_block[0, :] = bos_row

    # add perturbation
    position_block += positions_perturbation * torch.randn_like(position_block)

    # compose the block diagonal attention weights
    # compose the all-features block, acting on both informative and uninformative features
    all_features_block = torch.block_diag(info_block, noise_block)
    # add the perturbation
    if info_and_noninfo_perturbation > 0:
        all_features_block += info_and_noninfo_perturbation * torch.randn_like(all_features_block) / (3*partial_width)
    # compose witht he positions_block
    w_weights = torch.block_diag(all_features_block, position_block)
    # shape is [input_width, input_width], with input_width = 3*partial_width + (max_number_tokens + 1)

    # add a dummy head dimension
    w_weights = rearrange(w_weights, "i j -> 1 i j")
    # shape is [number_heads=1, input_width, input_width]

    return w_weights


def generate_attention_weights_markov_optionC(partial_width, max_number_tokens, shift=-1, version="same_token",
                                              informative_perturbation=0.0, positions_perturbation=0.0):
    """
    Returns an attention weight for the Markov chain task

    Parameters
    ----------
    partial_width: int
        Half the width of the informative features part of the input.
        For reference, the input is of size [informative_width + noninformative_width + one-hot_positions_width],
        with sizes:
            -> informative_width = 2*partial_width;
            -> noninformative_width = = 2*partial_width;
            -> one-hot_positions_width = max_number_tokens + 1 (the +1 is for the bos token)

    max_number_tokens: int
        Max sequence length allowed by the model.
        This defines the length of the one-hot positional encoding part of the input (see partial_width definition)

    shift: int
        Only relevant for the good heads "same_token" and "different_token". Defines the relative distance of the key
        token to which the query token attends to.
        The good heads have a shift of -/+ 1.

    version: str
        Can be one of the following:
            -> "same_token": good head, checks if the token at distance <shift> is the same as query, otherwise attends
            the bos token (i.e. nothing: bos token is zeros in the features space)
            -> "different_token": analogous to "same_token", but check is the key token is different than query
            -> "uniform attention": Attends all tokens equally. This is done by ignores the features space, and applying
            a uniform matrix of ones on the positions space.
            -> "blank": attention weights are zero. This can be used to generate random attention weights,
            by controlling the perturbation strength parameters.

    informative_perturbation: float
        adds a perturbation to the block of attention weights acting on the informative features subspace

    positions_perturbation: float
        adds a perturbation to the attention weights acting on the positional encoding subspace.
    """

    # version options: "same_token", "different_token", "uniform_attention", "blank".

    # Lorenzo: Here we define the parameters establishing the strength of the info and position part of the good head.
    # These are set to 1 without loss of generality, but I leave them here as a tunable parameter for completeness and
    # to match my notes on the derivation of the good head.
    informative_features_strength = 1.0
    position_strength = 1.0

    # NOTE: the returned w_weights should take key on the left and query on the right

    # generate the feature-informative block:
    if version == "same_token":
        info_block = torch.eye(2*partial_width)/partial_width
        # multiply by strength
        info_block *= informative_features_strength
    elif version == "different_token":
        id = torch.eye(partial_width)
        zeros = torch.zeros(partial_width, partial_width)
        lower_rectangle = torch.cat([id, zeros], dim=1)
        upper_rectangle = torch.cat([zeros, id], dim=1)
        info_block = torch.cat([upper_rectangle, lower_rectangle], dim=0)/partial_width
        # multiply by strength
        info_block *= informative_features_strength
    elif version == "blank" or version == "uniform_attention":
        info_block = torch.zeros(2*partial_width, 2*partial_width)/partial_width
    else:
        print("WARNING: the specified head version" + version + " is not valid. Exiting")
        sys.exit()
    # add perturbation
    if informative_perturbation > 0:
        info_block += informative_perturbation * torch.randn_like(info_block)/partial_width

    # generate the position block:
    if version == "blank":
        position_block = torch.zeros(max_number_tokens + 1, max_number_tokens + 1)
    elif version == "uniform_attention":
        position_block = torch.ones(max_number_tokens + 1, max_number_tokens + 1)
    else:  # "same_token", "different_token"
        # generate the position shift block
        offset = int(-1*shift)
        position_block = torch.diag_embed(torch.diagonal(torch.ones(max_number_tokens, max_number_tokens), offset=offset),
                                          offset=offset)
        # size [max_number_tokens, max_number_tokens]
        # multiply by strength
        position_block *= position_strength
        # add row-column for the bos position
        position_block = torch.block_diag(torch.zeros(1, 1), position_block)
        # size [max_number_tokens + 1, max_number_tokens + 1]
        # add attention to bos token from any other token (i.e. row [0, 1, 1, 1, 1, ..., 1])
        bos_strength = (position_strength + informative_features_strength +
                        max(position_strength, informative_features_strength))/2
        bos_row = torch.ones(max_number_tokens + 1) * bos_strength
        bos_row[0] = 0.0  # we don't want the bos token to attend to itself
        position_block[0, :] = bos_row

    # add perturbation
    if positions_perturbation > 0.0:
        position_block += positions_perturbation * torch.randn_like(position_block)

    # compose the block diagonal attention weights
    w_weights = torch.block_diag(info_block, position_block)
    # shape is [input_width, input_width], with input_width = 3*partial_width + (max_number_tokens + 1)

    # add a dummy head dimension
    w_weights = rearrange(w_weights, "i j -> 1 i j")
    # shape is [number_heads=1, input_width, input_width]

    return w_weights


def generate_attention_weights_markov_optionD(partial_width, number_noninformative_features, max_number_tokens,
                                              shift=-1, version="same_token",
                                              informative_perturbation=0.0, noninformative_perturbation=0.0,
                                              info_and_noninfo_perturbation=0.0, positions_perturbation=0.0,
                                              all_features_and_positions_off_diagonal_perturbation=0.0):
    """
    Returns an attention weight for the Markov chain task

    Parameters
    ----------
    partial_width: int
        Half the width of the informative features part of the input.
        For reference, the input is of size [informative_width + noninformative_width + one-hot_positions_width],
        with sizes:
            -> informative_width = 2*partial_width;
            -> noninformative_width = = 2*partial_width;
            -> one-hot_positions_width = max_number_tokens + 1 (the +1 is for the bos token)

    max_number_tokens: int
        Max sequence length allowed by the model.
        This defines the length of the one-hot positional encoding part of the input (see partial_width definition)

    shift: int
        Only relevant for the good heads "same_token" and "different_token". Defines the relative distance of the key
        token to which the query token attends to.
        The good heads have a shift of -/+ 1.

    version: str
        Can be one of the following:
            -> "same_token": good head, checks if the token at distance <shift> is the same as query, otherwise attends
            the bos token (i.e. nothing: bos token is zeros in the features space)
            -> "different_token": analogous to "same_token", but check is the key token is different than query
            -> "uniform attention": Attends all tokens equally. This is done by ignores the features space, and applying
            a uniform matrix of ones on the positions space.
            -> "blank": attention weights are zero. This can be used to generate random attention weights,
            by controlling the perturbation strength parameters.

    informative_perturbation: float
        adds a perturbation to the block of attention weights acting on the informative features subspace

    noninformative_perturbation: float
        adds a perturbation to the block of attention weights acting on the noninformative features subspace

    info_and_noninfo_perturbation: float
        adds a perturbation to the attention weights acting on the whole features subspace.
        The difference with perturbing just the informative and noninformative blocks separately, is that here the
        attention weights will also mix the informative and noninformative features subspace.
        NOTE: the positions subspace is never mixed with the others, instead. It is of a different nature (one-hot),
        than the featrures subsapce, so it should be treated separately.

    positions_perturbation: float
        adds a perturbation to the attention weights acting on the positional encoding subspace.
    """

    # version options: "same_token", "different_token", "uniform_attention", "blank".

    # Lorenzo: Here we define the parameters establishing the strength of the info and position part of the good head.
    # These are set to 1 without loss of generality, but I leave them here as a tunable parameter for completeness and
    # to match my notes on the derivation of the good head.
    informative_features_strength = 1.0
    position_strength = 1.0

    # NOTE: the returned w_weights should take key on the left and query on the right

    # generate the feature-informative block:
    if version == "same_token":
        info_block = torch.eye(2*partial_width)/partial_width
        # multiply by strength
        info_block *= informative_features_strength
    elif version == "different_token":
        id = torch.eye(partial_width)
        zeros = torch.zeros(partial_width, partial_width)
        lower_rectangle = torch.cat([id, zeros], dim=1)
        upper_rectangle = torch.cat([zeros, id], dim=1)
        info_block = torch.cat([upper_rectangle, lower_rectangle], dim=0)/partial_width
        # multiply by strength
        info_block *= informative_features_strength
    elif version == "blank" or version == "uniform_attention":
        info_block = torch.zeros(2*partial_width, 2*partial_width)/partial_width
    else:
        print("WARNING: the specified head version" + version + " is not valid. Exiting")
        sys.exit()
    # add perturbation
    info_block += informative_perturbation * torch.randn_like(info_block)/partial_width

    # generate the noise-features block:
    # same for all: "same_token", "different_token", "uniform_attention", "blank"
    if noninformative_perturbation > 0:
        noise_block = (noninformative_perturbation *
                       torch.randn(number_noninformative_features, number_noninformative_features)
                       / number_noninformative_features)
    else:
        noise_block = (torch.zeros(number_noninformative_features, number_noninformative_features)
                       / number_noninformative_features)

    # generate the position block:
    if version == "blank":
        position_block = torch.zeros(max_number_tokens + 1, max_number_tokens + 1)
    elif version == "uniform_attention":
        position_block = torch.ones(max_number_tokens + 1, max_number_tokens + 1)
    else:  # "same_token", "different_token"
        # generate the position shift block
        offset = int(-1*shift)
        position_block = torch.diag_embed(torch.diagonal(torch.ones(max_number_tokens, max_number_tokens), offset=offset),
                                          offset=offset)
        # size [max_number_tokens, max_number_tokens]
        # multiply by strength
        position_block *= position_strength
        # add row-column for the bos position
        position_block = torch.block_diag(torch.zeros(1, 1), position_block)
        # size [max_number_tokens + 1, max_number_tokens + 1]
        # add attention to bos token from any other token (i.e. row [0, 1, 1, 1, 1, ..., 1])
        bos_strength = (position_strength + informative_features_strength +
                        max(position_strength, informative_features_strength))/2
        bos_row = torch.ones(max_number_tokens + 1) * bos_strength
        bos_row[0] = 0.0  # we don't want the bos token to attend to itself
        position_block[0, :] = bos_row

    # add perturbation
    position_block += positions_perturbation * torch.randn_like(position_block)

    # compose the block diagonal attention weights
    # compose the all-features block, acting on both informative and uninformative features
    all_features_block = torch.block_diag(info_block, noise_block)
    # add the perturbation
    if info_and_noninfo_perturbation > 0:
        all_features_block += (info_and_noninfo_perturbation * torch.randn_like(all_features_block)
                               / (2*partial_width + number_noninformative_features))
    # compose with the positions_block
    w_weights = torch.block_diag(all_features_block, position_block)
    # shape is [input_width, input_width], with input_width = 3*partial_width + (max_number_tokens + 1)

    if all_features_and_positions_off_diagonal_perturbation > 0:
        # Create a random matrix
        random_matrix = torch.randn_like(w_weights) / (2*partial_width + number_noninformative_features)
        # Create a mask to identify off-diagonal blocks
        mask = 1 - torch.block_diag(torch.ones_like(all_features_block), torch.ones_like(position_block))
        # Zero out everything except the off-diagonal blocks
        random_matrix *= mask
        # add to the w_weights
        w_weights += all_features_and_positions_off_diagonal_perturbation * random_matrix

    # add a dummy head dimension
    w_weights = rearrange(w_weights, "i j -> 1 i j")
    # shape is [number_heads=1, input_width, input_width]

    return w_weights


def generate_attention_weights_markov_optionE(dataset_info_after_initialization,
                                              shift=-1, version="same_token",
                                              features_perturbation=0.0, positions_perturbation=0.0,
                                              features_positions_cross_perturbation=0.0):
    """
    Returns an attention weight for the Markov chain task

    Parameters
    ----------
    shift: int
        Only relevant for the good heads "same_token" and "different_token". Defines the relative distance of the key
        token to which the query token attends to.
        The good heads have a shift of -/+ 1.

    version: str
        Can be one of the following:
            -> "same_token": good head, checks if the token at distance <shift> is the same as query, otherwise attends
            the bos token (i.e. nothing: bos token is zeros in the features space)
            -> "different_token": analogous to "same_token", but check is the key token is different than query
            -> "uniform attention": Attends all tokens equally. This is done by ignores the features space, and applying
            a uniform matrix of ones on the positions space.
            -> "blank": attention weights are zero. This can be used to generate random attention weights,
            by controlling the perturbation strength parameters.

    features_perturbation: float
        adds a perturbation to the block of attention weights acting on the features subspace

    positions_perturbation: float
        adds a perturbation to the attention weights acting on the positional encoding subspace.

    features_positions_cross_perturbation: float
        adds a perturbation to the weights mixing features and positions in the attention matrix.
    """
    # version options: "same_token", "different_token", "uniform_attention", "blank".

    # Lorenzo: Here we define the parameters establishing the strength of the info and position part of the good head.
    # These are set to 1 without loss of generality, but I leave them here as a tunable parameter for completeness and
    # to match my notes on the derivation of the good head.
    informative_features_strength = 1.0
    position_strength = 1.0

    # NOTE: the returned w_weights should take key on the left and query on the right

    # collect info
    partial_width = dataset_info_after_initialization["partial_width"]
    max_number_tokens = dataset_info_after_initialization["max_number_tokens"]  # max sequence length
    v_minus = dataset_info_after_initialization["v_minus"]
    v_plus = dataset_info_after_initialization["v_plus"]

    # define the projection vector
    w_parallel = v_plus - v_minus
    # normalize
    q_parallel = w_parallel / (torch.dot(w_parallel, w_parallel)/2)

    # generate the feature-informative block:
    if version == "same_token":
        k_parallel = q_parallel
        info_block = einsum(k_parallel, q_parallel, "i, j -> i j")
        # multiply by strength
        info_block *= informative_features_strength
    elif version == "different_token":
        k_parallel = -1*q_parallel
        info_block = einsum(k_parallel, q_parallel, "i, j -> i j")
        # multiply by strength
        info_block *= informative_features_strength
    elif version == "blank" or version == "uniform_attention":
        info_block = torch.zeros(partial_width, partial_width)
    else:
        print("WARNING: the specified head version" + version + " is not valid. Exiting")
        sys.exit()
    # add perturbation
    if features_perturbation > 0:
        info_block += features_perturbation * torch.randn_like(info_block)/partial_width

    # generate the position block:
    if version == "blank":
        position_block = torch.zeros(max_number_tokens + 1, max_number_tokens + 1)
    elif version == "uniform_attention":
        position_block = torch.ones(max_number_tokens + 1, max_number_tokens + 1)
    else:  # "same_token", "different_token"
        # generate the position shift block
        offset = int(-1*shift)
        position_block = torch.diag_embed(torch.diagonal(torch.ones(max_number_tokens, max_number_tokens), offset=offset),
                                          offset=offset)
        # size [max_number_tokens, max_number_tokens]
        # multiply by strength
        position_block *= position_strength
        # add row-column for the bos position
        position_block = torch.block_diag(torch.zeros(1, 1), position_block)
        # size [max_number_tokens + 1, max_number_tokens + 1]
        # add attention to bos token from any other token (i.e. row [0, 1, 1, 1, 1, ..., 1])
        bos_strength = (position_strength + informative_features_strength +
                        max(position_strength-informative_features_strength, informative_features_strength))/2
        bos_row = torch.ones(max_number_tokens + 1) * bos_strength
        bos_row[0] = 0.0  # we don't want the bos token to attend to itself
        position_block[0, :] = bos_row

    # add perturbation
    position_block += positions_perturbation * torch.randn_like(position_block)

    # compose with the positions_block
    w_weights = torch.block_diag(info_block, position_block)
    # shape is [input_width, input_width], with input_width = partial_width + (max_number_tokens + 1)

    if features_positions_cross_perturbation > 0:
        # Create a random matrix
        random_matrix = torch.randn_like(w_weights) / np.sqrt(partial_width)
        # Create a mask to identify off-diagonal blocks
        mask = 1 - torch.block_diag(torch.ones_like(info_block), torch.ones_like(position_block))
        # Zero out everything except the off-diagonal blocks
        random_matrix *= mask
        # add to the w_weights
        w_weights += features_perturbation * random_matrix

    # add a dummy head dimension
    w_weights = rearrange(w_weights, "i j -> 1 i j")
    # shape is [number_heads=1, input_width, input_width]

    return w_weights


def mini_position_shift_matrix(shift, pos_encoding_period=1.0e4):
    freq = 2*np.pi/pos_encoding_period
    x = freq*shift
    pos_shift_matrix = torch.tensor([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])
    return pos_shift_matrix


# PREPARING DATASET FUNCTIONS


def prepare_dataset(dataset_location, dataset_info, train=True, loginf=None):
    # NOTE: dataset_info contains the user-given dataset info.
    # If these are incompatible, they will be changed to default values, but the value in dataset_info won't be changed.
    # This anyway ensures the dataset is retrievable, without risking of overwriting stuff.

    # TO PUT INTO DOCUMENTATION:
    # this function always returns: data, labels, dataset_info
    # data is of size [number_examples, input_width, number_tokens]
    # labels depends on the task, but typical task is binary regression, for which we have size [number_examples]

    if dataset_info["dataset"] == "MNIST_binary_regression":
        return prepare_dataset_MNIST_binary_regression(dataset_location, dataset_info, train=train)

    if dataset_info["dataset"] == "CIFAR10_binary_regression":
        return prepare_dataset_CIFAR10_binary_regression(dataset_location, dataset_info, train=train)

    if "pretrained" in dataset_info["dataset"]:
        if "incontext" in dataset_info["dataset"]:
            if train is True:
                return prepare_dataset_pretrained_optionA(dataset_location, dataset_info, train=train)
            else:
                return prepare_dataset_pretrained_optionA(
                    dataset_location, dataset_info, train=train, incontext=True)
        else:
            return prepare_dataset_pretrained_optionA(dataset_location, dataset_info, train=train, loginf=loginf)

    if dataset_info["dataset"] == "markov_optionA":
        return prepare_dataset_markov_optionA(dataset_info, train=train)

    if dataset_info["dataset"] == "markov_optionB":
        return prepare_dataset_markov_optionB(dataset_info, train=train)

    if dataset_info["dataset"] == "markov_optionC":
        return prepare_dataset_markov_optionC(dataset_info, train=train)

    if dataset_info["dataset"] == "markov_optionD":
        return prepare_dataset_markov_optionD(dataset_info, train=train)

    if dataset_info["dataset"] == "markov_optionE":
        return prepare_dataset_markov_optionE(dataset_info, train=train)


def prepare_dataset_markov_optionE(dataset_info, train=True, return_states=False):

    print("\nINITIALIZE MARKOV DATASET: START\n")

    if train:
        seed_everything(seed=27)
    else:
        seed_everything(seed=59)

    # collect info
    p_a_plus = dataset_info["p_a_plus"]
    p_a_minus = dataset_info["p_a_minus"]
    p_b_plus = dataset_info["p_b_plus"]
    p_b_minus = dataset_info["p_b_minus"]
    number_examples = dataset_info["number_examples"]
    number_tokens = dataset_info["number_tokens"]
    partial_width = dataset_info["partial_width"]
    max_number_tokens = dataset_info["max_number_tokens"]  # max sequence length
    perpendicular_noise_strength = dataset_info["perpendicular_noise_strength"]
    parallel_noise_strength = dataset_info["parallel_noise_strength"]
    out_of_plane_noise_strength = dataset_info["out_of_plane_noise_strength"]
    v_minus = dataset_info["v_minus"]
    v_plus = dataset_info["v_plus"]

    # derive number examples per class
    number_examples_per_class = int(number_examples // 2)
    # NOTE: number_examples stores the user specified number of examples. If this is not a multiple of
    # tot_number_labels, this will not represent the actual number of examples (a.k.a. P).
    # Always refer to the training_data size to extract the real number of examples (as the ConvergentSummationHeads
    # class also does)

    # generate the labels
    labels = torch.cat([1.0*torch.ones(number_examples_per_class), -1.0*torch.ones(number_examples_per_class)])

    # generate the v_plus and v_minus states, if not given
    if (v_plus is None) or (v_minus is None):
        # note: the sqrt(2) is there such that v_plus and v_minus have square norm of partial_width
        v_plus = torch.cat([torch.ones(int(partial_width/2)), torch.zeros(int(partial_width/2))]) * np.sqrt(2.0)
        v_minus = torch.cat([torch.zeros(int(partial_width/2)), torch.ones(int(partial_width/2))]) * np.sqrt(2.0)
    else:
        print("Initializing user-defined feature vectors v_plus and v_minus. "
              "Please check the norm^2/partial_width is of order 1 and the same for both vectors")
        print(f"norm^2/partial_width v_plus = {torch.dot(v_plus, v_plus)/partial_width}")
        print(f"norm^2/partial_width v_minus = {torch.dot(v_minus, v_minus) / partial_width}")
    # update v_plus and v_minus states in dataset_info
    dataset_info["v_minus"] = v_minus
    v_plus = dataset_info["v_plus"] = v_plus

    # define the normalized w_perpendicular and w_parallel vectors
    w_perpendicular = v_plus - v_minus
    w_perpendicular_direction = w_perpendicular / torch.sqrt(torch.dot(w_perpendicular, w_perpendicular))
    w_parallel = v_plus + v_minus
    w_parallel_direction = w_parallel / torch.sqrt(torch.dot(w_parallel, w_parallel))

    # generate the features part of the examples
    feature_examples, state_examples = generate_markov_chain(p_a_plus, p_a_minus, p_b_plus, p_b_minus,
                                                             number_examples_per_class, number_tokens, partial_width,
                                                             features_perturbation_strength=0.0,
                                                             flip_v_minus=False,
                                                             v_plus=v_plus, v_minus=v_minus)
    # feature_examples is of size [# examples, partial_width, # tokens]
    # state_examples is of size [# examples, 1, # tokens]

    # add the noise
    isotropic_noise = torch.randn_like(feature_examples)  # size [# examples, partial_width, # tokens]
    # project along parallel direction
    parallel_noise = einsum(w_parallel_direction, w_parallel_direction, isotropic_noise,
                            "width2, width1, examples width1 tokens -> examples width2 tokens")
    # project along perpendicular direction
    perpendicular_noise = einsum(w_perpendicular_direction, w_perpendicular_direction, isotropic_noise,
                                 "width2, width1, examples width1 tokens -> examples width2 tokens")
    # project along out_of_plane directions
    out_of_plane_noise = isotropic_noise - parallel_noise - perpendicular_noise
    # combine all noises with their strength
    noise = (parallel_noise_strength*parallel_noise + perpendicular_noise_strength*perpendicular_noise
             + out_of_plane_noise_strength*out_of_plane_noise)
    # add noise to feature examples
    feature_examples += noise  # size [# examples, partial_width, # tokens]

    # generate the bos tokens (one for each example)
    bos_tokens = torch.zeros(feature_examples.size()[0], feature_examples.size()[1], 1)
    # size  # size [# examples, partial_width, 1]
    # concatenate to the tokens (along the token direction)
    feature_examples = torch.cat([bos_tokens, feature_examples], dim=2)
    # size: [# examples, partial_width, # tokens + 1]

    # generate the positional encoding (for a single example)
    pos_encoding = torch.eye(max_number_tokens + 1)
    # extract only up to the sequence length:
    pos_encoding = pos_encoding[:, 0:(number_tokens + 1)]  # size: [width: max_number_tokens + 1, length: # tokens + 1]
    # repeat across examples
    positional_encodings = repeat(pos_encoding, 'width tokens-> examples width tokens',
                                  examples=feature_examples.size()[0])
    # size: [# examples, width: max_number_tokens + 1, length: # tokens + 1]
    # concatenate to the examples (along the features direction)
    feature_examples = torch.cat([feature_examples, positional_encodings], dim=1)

    # unseed everything
    unseed_everything()

    if return_states:
        return feature_examples, labels, dataset_info, state_examples
    else:
        return feature_examples, labels, dataset_info


def prepare_dataset_markov_optionD(dataset_info, train=True, return_states=False):
    if train:
        seed_everything(seed=27)
    else:
        seed_everything(seed=59)

    # collect info
    p_a_plus = dataset_info["p_a_plus"]
    p_a_minus = dataset_info["p_a_minus"]
    p_b_plus = dataset_info["p_b_plus"]
    p_b_minus = dataset_info["p_b_minus"]
    number_examples = dataset_info["number_examples"]
    number_tokens = dataset_info["number_tokens"]
    partial_width = dataset_info["partial_width"]
    number_noninformative_features = dataset_info["number_noninformative_features"]
    noninformative_features_strength = torch.tensor(dataset_info["noninformative_features_strength"])
    max_number_tokens = dataset_info["max_number_tokens"]  # max sequence length
    features_perturbation_strength = dataset_info["features_perturbation_strength"]
    flip_v_minus = dataset_info["flip_v_minus"]
    v_minus = dataset_info["v_minus"]
    v_plus = dataset_info["v_plus"]

    # derive number examples per class
    number_examples_per_class = int(number_examples // 2)
    # NOTE: number_examples stores the user specified number of examples. If this is not a multiple of
    # tot_number_labels, this will not represent the actual number of examples (a.k.a. P).
    # Always refer to the training_data size to extract the real number of examples (as the ConvergentSummationHeads
    # class also does)

    # generate the labels
    labels = torch.cat([1.0*torch.ones(number_examples_per_class), -1.0*torch.ones(number_examples_per_class)])

    # generate the features part of the examples
    feature_examples, state_examples = generate_markov_chain(p_a_plus, p_a_minus, p_b_plus, p_b_minus,
                                                             number_examples_per_class, number_tokens, partial_width,
                                                             features_perturbation_strength=
                                                             features_perturbation_strength,
                                                             flip_v_minus=flip_v_minus,
                                                             v_plus=v_plus, v_minus=v_minus)
    # feature_examples is of size [# examples, 2*partial_width, # tokens]
    # state_examples is of size [# examples, 1, # tokens]

    # generate the random features part of the examples
    random_features = (noninformative_features_strength *
                       torch.randn(feature_examples.size()[0], number_noninformative_features, number_tokens))
    # concatenate the random features part of the examples
    feature_examples = torch.cat([feature_examples, random_features], dim=1)
    # SIZE: [# examples, 2*partial_width + number_noninformative_features, # tokens]

    # generate the bos tokens (one for each example)
    bos_tokens = torch.zeros(feature_examples.size()[0], 2*partial_width + number_noninformative_features, 1)
    # size [# examples, 2*partial_width + number_noninformative_features, 1]
    # concatenate to the tokens (along the token direction)
    feature_examples = torch.cat([bos_tokens, feature_examples], dim=2)
    # size: [# examples, 3*partial_width, # tokens + 1]

    # generate the positional encoding (for a single example)
    pos_encoding = torch.eye(max_number_tokens + 1)
    # extract only up to the sequence length:
    pos_encoding = pos_encoding[:, 0:(number_tokens + 1)]  # size: [width: max_number_tokens + 1, length: # tokens + 1]
    # repeat across examples
    positional_encodings = repeat(pos_encoding, 'width tokens-> examples width tokens',
                                  examples=feature_examples.size()[0])
    # size: [# examples, width: max_number_tokens + 1, length: # tokens + 1]
    # concatenate to the examples (along the features direction)
    feature_examples = torch.cat([feature_examples, positional_encodings], dim=1)

    # unseed everything
    unseed_everything()

    if return_states:
        return feature_examples, labels, dataset_info, state_examples
    else:
        return feature_examples, labels, dataset_info


def prepare_dataset_markov_optionC(dataset_info, train=True, return_states=False):
    # OPTIONAL TODO: add possibility of generic v_plus and v_minus. I.e. informative part would project along these
    # two vectors
    # set different seed for Train/Test;
    if train:
        seed_everything(seed=27)
    else:
        seed_everything(seed=59)

    # collect info
    p_a_plus = dataset_info["p_a_plus"]
    p_a_minus = dataset_info["p_a_minus"]
    p_b_plus = dataset_info["p_b_plus"]
    p_b_minus = dataset_info["p_b_minus"]
    number_examples = dataset_info["number_examples"]
    number_tokens = dataset_info["number_tokens"]
    partial_width = dataset_info["partial_width"]
    max_number_tokens = dataset_info["max_number_tokens"]  # max sequence length
    features_perturbation_strength = dataset_info["features_perturbation_strength"]
    flip_v_minus = dataset_info["flip_v_minus"]
    v_minus = dataset_info["v_minus"]
    v_plus = dataset_info["v_plus"]

    # derive number examples per class
    number_examples_per_class = int(number_examples // 2)
    # NOTE: number_examples stores the user specified number of examples. If this is not a multiple of
    # tot_number_labels, this will not represent the actual number of examples (a.k.a. P).
    # Always refer to the training_data size to extract the real number of examples (as the ConvergentSummationHeads
    # class also does)

    # generate the labels
    labels = torch.cat([1.0*torch.ones(number_examples_per_class), -1.0*torch.ones(number_examples_per_class)])

    # generate the features part of the examples
    feature_examples, state_examples = generate_markov_chain(p_a_plus, p_a_minus, p_b_plus, p_b_minus,
                                                             number_examples_per_class, number_tokens, partial_width,
                                                             features_perturbation_strength=
                                                             features_perturbation_strength,
                                                             flip_v_minus=flip_v_minus,
                                                             v_plus=v_plus, v_minus=v_minus)
    # feature_examples is of size [# examples, 2*partial_width, # tokens]
    # state_examples is of size [# examples, 1, # tokens]

    # generate the bos tokens (one for each example)
    bos_tokens = torch.zeros(feature_examples.size()[0], 2 * partial_width, 1)  # size [# examples, 2*partial_width, 1]
    # concatenate to the tokens (along the token direction)
    feature_examples = torch.cat([bos_tokens, feature_examples], dim=2)
    # size: [# examples, 3*partial_width, # tokens + 1]

    # generate the positional encoding (for a single example)
    pos_encoding = torch.eye(max_number_tokens + 1)
    # extract only up to the sequence length:
    pos_encoding = pos_encoding[:, 0:(number_tokens + 1)]  # size: [width: max_number_tokens + 1, length: # tokens + 1]
    # repeat across examples
    positional_encodings = repeat(pos_encoding, 'width tokens-> examples width tokens',
                                  examples=feature_examples.size()[0])
    # size: [# examples, width: max_number_tokens + 1, length: # tokens + 1]
    # concatenate to the examples (along the features direction)
    feature_examples = torch.cat([feature_examples, positional_encodings], dim=1)

    # unseed everything
    unseed_everything()

    if return_states:
        return feature_examples, labels, dataset_info, state_examples
    else:
        return feature_examples, labels, dataset_info


def prepare_dataset_markov_optionB(dataset_info, train=True, return_states=False):
    # set different seed for Train/Test;
    if train:
        seed_everything(seed=27)
    else:
        seed_everything(seed=59)

    # collect info
    p_a_plus = dataset_info["p_a_plus"]
    p_a_minus = dataset_info["p_a_minus"]
    p_b_plus = dataset_info["p_b_plus"]
    p_b_minus = dataset_info["p_b_minus"]
    number_examples = dataset_info["number_examples"]
    number_tokens = dataset_info["number_tokens"]
    partial_width = dataset_info["partial_width"]
    max_number_tokens = dataset_info["max_number_tokens"]  # max sequence length
    features_perturbation_strength = dataset_info["features_perturbation_strength"]

    # derive number examples per class
    number_examples_per_class = int(number_examples // 2)
    # NOTE: number_examples stores the user specified number of examples. If this is not a multiple of
    # tot_number_labels, this will not represent the actual number of examples (a.k.a. P).
    # Always refer to the training_data size to extract the real number of examples (as the ConvergentSummationHeads
    # class also does)

    # generate the labels
    labels = torch.cat([1.0*torch.ones(number_examples_per_class), -1.0*torch.ones(number_examples_per_class)])

    # generate the features part of the examples
    feature_examples, state_examples = generate_markov_chain(p_a_plus, p_a_minus, p_b_plus, p_b_minus,
                                                             number_examples_per_class, number_tokens, partial_width,
                                                             features_perturbation_strength=
                                                             features_perturbation_strength)
    # feature_examples is of size [# examples, 2*partial_width, # tokens]
    # state_examples is of size [# examples, 1, # tokens]

    # generate the random features part of the examples
    random_features = torch.randn(feature_examples.size()[0], partial_width, number_tokens)
    # concatenate the random features part of the examples
    feature_examples = torch.cat([feature_examples, random_features], dim=1)
    # SIZE: [# examples, 3*partial_width, # tokens]

    # generate the bos tokens (one for each example)
    bos_tokens = torch.zeros(feature_examples.size()[0], 3 * partial_width, 1)  # size [# examples, 3*partial_width, 1]
    # concatenate to the tokens (along the token direction)
    feature_examples = torch.cat([bos_tokens, feature_examples], dim=2)
    # size: [# examples, 3*partial_width, # tokens + 1]

    # generate the positional encoding (for a single example)
    pos_encoding = torch.eye(max_number_tokens + 1)
    # extract only up to the sequence length:
    pos_encoding = pos_encoding[:, 0:(number_tokens + 1)]  # size: [width: max_number_tokens + 1, length: # tokens + 1]
    # repeat across examples
    positional_encodings = repeat(pos_encoding, 'width tokens-> examples width tokens',
                                  examples=feature_examples.size()[0])
    # size: [# examples, width: max_number_tokens + 1, length: # tokens + 1]
    # concatenate to the examples (along the features direction)
    feature_examples = torch.cat([feature_examples, positional_encodings], dim=1)

    # unseed everything
    unseed_everything()

    if return_states:
        return feature_examples, labels, dataset_info, state_examples
    else:
        return feature_examples, labels, dataset_info


def prepare_dataset_markov_optionA(dataset_info, train=True, return_states=False):
    # set different seed for Train/Test;
    if train:
        seed_everything(seed=27)
    else:
        seed_everything(seed=59)

    # collect info
    p_a_plus = dataset_info["p_a_plus"]
    p_a_minus = dataset_info["p_a_minus"]
    p_b_plus = dataset_info["p_b_plus"]
    p_b_minus = dataset_info["p_b_minus"]
    number_examples = dataset_info["number_examples"]
    number_tokens = dataset_info["number_tokens"]
    partial_width = dataset_info["partial_width"]
    pos_encoding_period = dataset_info["positional_encoding_period"]
    features_perturbation_strength = dataset_info["features_perturbation_strength"]

    # derive number examples per class
    number_examples_per_class = int(number_examples // 2)
    # NOTE: number_examples stores the user specified number of examples. If this is not a multiple of
    # tot_number_labels, this will not represent the actual number of examples (a.k.a. P).
    # Always refer to the training_data size to extract the real number of examples (as the ConvergentSummationHeads
    # class also does)

    # generate the labels
    labels = torch.cat([1.0*torch.ones(number_examples_per_class), -1.0*torch.ones(number_examples_per_class)])

    # generate the features part of the examples
    feature_examples, state_examples = generate_markov_chain(p_a_plus, p_a_minus, p_b_plus, p_b_minus,
                                                             number_examples_per_class, number_tokens, partial_width,
                                                             features_perturbation_strength=
                                                             features_perturbation_strength)
    # feature_examples is of size [# examples, 2*partial_width, # tokens]
    # state_examples is of size [# examples, 1, # tokens]

    # generate the random features part of the examples
    random_features = torch.randn(feature_examples.size()[0], partial_width, number_tokens)
    # concatenate the random features part of the examples
    feature_examples = torch.cat([feature_examples, random_features], dim=1)

    # generate the "not_bof_token" feature
    not_bos_feature = torch.tensor([0., 1.])
    # repeat across tokens and across examples
    not_bos_feature = repeat(not_bos_feature, 'width -> examples width tokens',
                             examples=feature_examples.size()[0], tokens=number_tokens)
    # concatenate it to the tokens
    feature_examples = torch.cat([feature_examples, not_bos_feature], dim=1)

    # generate the positional encoding feature
    positional_encoding = mini_positional_encoding(number_tokens, period=pos_encoding_period)
    # repeat across examples
    positional_encodings = repeat(positional_encoding, 'width tokens-> examples width tokens',
                                  examples=feature_examples.size()[0])
    # concatenate it to the tokens
    feature_examples = torch.cat([feature_examples, positional_encodings], dim=1)

    # generate the bos token
    bos_token = torch.cat([torch.zeros(3 * partial_width), torch.tensor([1.0, 0.0, 0.0, 0.0])])
    # repeat across across examples
    bos_tokens = repeat(bos_token, 'width-> examples width',
                        examples=feature_examples.size()[0])
    # add a dummy token direction
    bos_tokens = rearrange(bos_tokens, "b i -> b i 1")
    # concatenate to the tokens (along the token direction)
    feature_examples = torch.cat([bos_tokens, feature_examples], dim=2)
    # feature_examples is of size [# examples, input_width = 3*partial_width + 4, # tokens]
    # state_examples is of size [# examples, 1, # tokens]

    # unseed everything
    unseed_everything()

    if return_states:
        return feature_examples, labels, dataset_info, state_examples
    else:
        return feature_examples, labels, dataset_info


def generate_markov_chain(p_a_plus, p_a_minus, p_b_plus, p_b_minus, number_examples_per_class, number_tokens,
                          partial_width, features_perturbation_strength=0.0, flip_v_minus=False, v_plus=None,
                          v_minus=None):
    device = "cpu"  # random number generation always on cpu!

    if (v_plus is None) or (v_minus is None):
        v_plus = torch.cat([torch.ones(partial_width), torch.zeros(partial_width)])
        v_minus = torch.cat([torch.zeros(partial_width), torch.ones(partial_width)])
    else:
        v_plus = v_plus
        v_minus = v_minus
    if flip_v_minus:
        v_minus = -1.0*v_minus

    feature_examples = []
    state_examples = []
    # class a
    p_pluses = [p_a_plus, p_b_plus]
    p_minuses = [p_a_minus, p_b_minus]
    for (p_plus, p_minus) in zip(p_pluses, p_minuses):
        for p in range(number_examples_per_class):
            # generate the first state
            coin = torch.rand(1, device=device)
            if coin >= 0.5:
                state = torch.tensor([1])
                feature = v_plus.clone()
            else:
                state = torch.tensor([-1])
                feature = v_minus.clone()

            # add a token dimension
            state = rearrange(state, "i -> i 1")
            feature = rearrange(feature, "i -> i 1")

            for t in range(number_tokens - 1):
                # select the right probability given the current state
                # select the stay/leave states given the current state
                if state[0, -1] == 1:
                    prob = p_plus
                    stay_feature = rearrange(v_plus.clone(), "i -> i 1")
                    leave_feature = rearrange(v_minus.clone(), "i -> i 1")
                    stay_state = rearrange(torch.tensor([1]), "i -> i 1")
                    leave_state = rearrange(torch.tensor([-1]), "i -> i 1")
                else:
                    prob = p_minus
                    stay_feature = rearrange(v_minus.clone(), "i -> i 1")
                    leave_feature = rearrange(v_plus.clone(), "i -> i 1")
                    stay_state = rearrange(torch.tensor([-1]), "i -> i 1")
                    leave_state = rearrange(torch.tensor([1]), "i -> i 1")
                # flip a coin and produce new state
                coin = torch.rand(1, device=device)
                if coin < prob:
                    # stay in the same state
                    feature = torch.cat([feature, stay_feature], dim=1)
                    state = torch.cat([state, stay_state], dim=1)
                else:
                    # change state
                    feature = torch.cat([feature, leave_feature], dim=1)
                    state = torch.cat([state, leave_state], dim=1)
            feature_examples.append(feature)
            state_examples.append(state)

    # stack all examples along a new batch dimension
    feature_examples = torch.stack(feature_examples, dim=0)  # size [# examples, 2*partial_width, # tokens]
    state_examples = torch.stack(state_examples, dim=0)  # size [# examples, 1, # tokens]

    # add noise to features
    if features_perturbation_strength > 0.0:
        feature_examples += features_perturbation_strength * torch.randn_like(feature_examples)

    return feature_examples, state_examples


def mini_positional_encoding(number_tokens, period=1.0e4):
    # note: if the number of tokens varies across examples, just generate this mini positional encoding with the
    # largest value of number of tokens. Then attach to each example the pos_encoding only up to the number of tokens
    # needed
    positions = torch.arange(number_tokens)
    frequency = 2.0*torch.pi / period
    cosines = torch.cos(frequency*positions)
    sines = torch.sin(frequency*positions)

    pos_encoding = torch.stack([cosines, sines], dim=0)  # size [2, # number tokens]

    return pos_encoding


def prepare_dataset_pretrained_optionA(dataset_location, dataset_info, train=True, incontext=False, loginf=None):
    data = torch.load(dataset_location + "/pretrained_heads/" + dataset_info["dataset"] + "/"
                      + dataset_info["dataset"] + ".pt",
                      map_location=torch.device('cpu'))  # we load everything on cpu, as usual
    _print = loginf if loginf is not None else print
    _print("\nPREPARING DATASET: START")
    _print("\n")
    _print("dataset: " + dataset_info["dataset"])

    # collect the train/test dataset
    if train:
        input = data['x_zeros_train']
        labels = data['y_labels_train']
    else:
        input = data['x_zeros_test']
        labels = data['y_labels_test']
        if incontext:
            _print("Using extra datasets for incontext learning")
            # mark the incontext eval case into `dataset_info`
            dataset_info["extra_testsets"] = ['mnist', 'fashion', 'cifar']
            extra_input = {}
            extra_labels = {}
            for extra_data in ['mnist', 'fashion', 'cifar']:
                extra_input[extra_data] = data[f'x_zeros_test_{extra_data}'].type(torch.get_default_dtype())
                extra_labels[extra_data] = data[f'y_labels_test_{extra_data}'].type(torch.get_default_dtype())
    do_extra_incontext_eval = (not train) and incontext
    # collect the query and key weights
    q_weights = data["w_q_weigts"]
    k_weights = data["w_k_weigts"]
    # convert everything to default dtype
    input = input.type(torch.get_default_dtype())
    labels = labels.type(torch.get_default_dtype())
    if type(q_weights) is list:
        q_new_list = []
        for q_w in q_weights:
            q_new_list.append(q_w.type(torch.get_default_dtype()))
        q_weights = q_new_list
    else:
        q_weights = q_weights.type(torch.get_default_dtype())
    if type(k_weights) is list:
        k_new_list = []
        for k_w in k_weights:
            k_new_list.append(k_w.type(torch.get_default_dtype()))
        k_weights = k_new_list
    else:
        k_weights = k_weights.type(torch.get_default_dtype())
    # collect and print information
    maximum_number_examples = input.size()[0]
    number_tokens = input.size()[1]
    if type(q_weights) is list:
        qk_internal_dimension = q_weights[0].shape[2]
        input_width = q_weights[0].size()[1]
        number_attention_layers = len(q_weights)
        number_heads = []
        for i in range(number_attention_layers):
            number_heads.append(q_weights[i].shape[0])
    else:
        qk_internal_dimension = q_weights.size()[3]
        input_width = q_weights.size()[2]
        number_heads = q_weights.size()[1]
        number_attention_layers = q_weights.size()[0]
    dataset_info["number_tokens"] = number_tokens
    dataset_info["maximum_number_examples"] = maximum_number_examples
    dataset_info["qk_internal_dimension"] = qk_internal_dimension
    dataset_info["input_width"] = input_width
    dataset_info["number_heads"] = number_heads
    dataset_info["number_attention_layers"] = number_attention_layers
    if dataset_info["dataset"] == "feb22_1v1_v0_pretrained":
        print("info: pretrained heads on CIFAR10 binary regression task. "
              "The two classes are two single categories from CIFAR10")
    _print(f"number of tokens: {number_tokens}")
    _print(f"number of layers: {number_attention_layers}")
    _print(f"number of heads: {number_heads}")
    _print(f"input width: {input_width}")
    _print(f"qk internal dimension: {qk_internal_dimension}")

    # order by label and extract only used examples
    number_examples_per_label = int(dataset_info["number_examples"] // 2)
    # label 1
    indices = labels == 1.0
    input_class_a = input[indices]
    input_class_a = input_class_a[0:number_examples_per_label]
    labels_class_a = labels[indices]
    labels_class_a = labels_class_a[0:number_examples_per_label]
    # label -1
    indices = labels == -1.0
    input_class_b = input[indices]
    input_class_b = input_class_b[0:number_examples_per_label]
    labels_class_b = labels[indices]
    labels_class_b = labels_class_b[0:number_examples_per_label]

    # rejoin label-ordered data
    input_ordered = torch.cat((input_class_a, input_class_b))  # size: [# examples, # tokens, input_width]
    labels_ordered = torch.cat((labels_class_a, labels_class_b))  # size: [# examples]

    ## any benefits for re-ordering test samples?
    ## for now just following what's done above
    if do_extra_incontext_eval:
        extra_input_ordered = {}
        extra_labels_ordered = {}
        for extra_data in dataset_info["extra_testsets"]:
            input = extra_input[extra_data]
            labels = extra_labels[extra_data]
            # label 1
            indices = labels == 1.0
            input_class_a = input[indices]
            input_class_a = input_class_a[0:number_examples_per_label]
            labels_class_a = labels[indices]
            labels_class_a = labels_class_a[0:number_examples_per_label]
            # label -1
            indices = labels == -1.0
            input_class_b = input[indices]
            input_class_b = input_class_b[0:number_examples_per_label]
            labels_class_b = labels[indices]
            labels_class_b = labels_class_b[0:number_examples_per_label]

            # rejoin label-ordered data
            extra_input_ordered[extra_data] = torch.cat((input_class_a, input_class_b))  # size: [# examples, # tokens, input_width]
            extra_labels_ordered[extra_data] = torch.cat((labels_class_a, labels_class_b))  # size: [# examples]

            extra_input_ordered[extra_data] = rearrange(
                extra_input_ordered[extra_data], "examples tokens width -> examples width tokens")

    # print further info
    number_used_examples = labels_ordered.size()[0]
    _print(f"maximum number examples: {maximum_number_examples}")
    _print(f"number of used examples: {number_used_examples}")

    # rearrange indices into the standard ordering [# examples, input_width, # tokens]
    input_ordered = rearrange(input_ordered, "examples tokens width -> examples width tokens")

    """
    rearrange q/k weights in the standard way:
    list of length L, with L the number of attention layers, i.e. a list [Q/K1, Q/K2, ..., Q/KL]
    each weight in the list is a torch.tensor of size [number_heads at layer, qk_internal_dimension, input_width]
    currently they are of size [# layers, # number heads, input_width, qk_internal_dimension]
    so we need to flip the last two axis
    For the normalization we are fine:
    the standard normalization is 1/sqrt(input_width).
    Here they are also normalized by 1/sqrt(input_width) 
    (they are normalized like this at initialization prior to learning)
    """
    query_weights = []
    key_weights = []
    for l in range(number_attention_layers):
        # extract weights at given layer
        q_w = q_weights[l]
        k_w = k_weights[l]
        # rearrange
        q_w = rearrange(q_w, "head width internal -> head internal width")
        k_w = rearrange(k_w, "head width internal -> head internal width")
        # uncomment to check mean and std
        _print(torch.std(q_w)*np.sqrt(q_w.size()[-1]))
        _print(torch.std(k_w)*np.sqrt(k_w.size()[-1]))
        _print(torch.mean(q_w)*np.sqrt(q_w.size()[-1]))
        _print(torch.mean(k_w)*np.sqrt(k_w.size()[-1]))
        query_weights.append(q_w)
        key_weights.append(k_w)

    dataset_info["query_weights"] = query_weights
    dataset_info["key_weights"] = key_weights

    _print("\nPREPARING DATASET: END")
    _print("\n")

    if do_extra_incontext_eval:
        input_ordered = (input_ordered, extra_input_ordered)
        labels_ordered = (labels_ordered, extra_labels_ordered)

    return input_ordered, labels_ordered, dataset_info


def prepare_dataset_MNIST_binary_regression(dataset_location, dataset_info, train=True,
                                            mean=33.5153694152832, std=78.7578353881836):

    print("\nPREPARING DATASET: START")
    print("\n")
    print("dataset: MNIST_binary_regression")

    # here we do everything on RAM and CPU
    device = "cpu"

    # seed everything so we are sure the date is always loaded in the same way
    seed_everything(seed=1)

    # RETRIEVE DATASET INFO
    labels_class_a = dataset_info["labels_class_a"]
    number_labels_class_a = len(labels_class_a)
    labels_class_b = dataset_info["labels_class_b"]
    number_labels_class_b = len(labels_class_b)
    tot_number_labels = number_labels_class_a + number_labels_class_b
    # NOTE: number_examples stores the user specified number of examples. If this is not a multiple of
    # tot_number_labels, this will not represent the actual number of examples (a.k.a. P).
    # Always refer to the training_data size to extract the real number of examples (as the ConvergentSummationHeads
    # class also does)
    number_examples = dataset_info["number_examples"]
    number_examples_per_label = int(number_examples // tot_number_labels)
    patch_linear_size = dataset_info["patch_linear_size"]

    # LOAD DATA
    training_data = datasets.MNIST(
        root=dataset_location,
        train=train,
        download=True,
        transform=ToTensor()
    )
    if 28 % patch_linear_size != 0:
        print(
            f"the chose patch linear size of {patch_linear_size} is not a dividend of 28. Using a patch linear size "
            f"of 4 instead")
        patch_linear_size = 4

    # DIVIDE DATA INTO CLASSES A AND B
    # CLASS A
    class_a_images = []
    for label in labels_class_a:
        # exctract images with the given label
        indices = training_data.targets == label
        images = training_data.data[indices]  # size: [# examples per label, image height, image width]

        # extract only a number "number_examples_per_class" of examples
        images = images[0:number_examples_per_label]

        # images are loaded with type ByteTensor. Convert them to type Float of default type
        # (whatever it is chosen in main, float32 or float64)
        images = images.type(torch.get_default_dtype()).to(device)

        class_a_images.append(images)
    class_a_images = torch.cat(class_a_images, dim=0)  # size [# examples per class, image height, image width]

    # CLASS B
    class_b_images = []
    for label in labels_class_b:
        # exctract images with the given label
        indices = training_data.targets == label
        images = training_data.data[indices]  # size: [# examples per label, image height, image width]

        # extract only a number "number_examples_per_class" of examples
        images = images[0:number_examples_per_label]

        # images are loaded with type ByteTensor. Convert them to type Float of default type
        # (whatever it is chosen in main, float32 or float64)
        images = images.type(torch.get_default_dtype()).to(device)

        class_b_images.append(images)
    class_b_images = torch.cat(class_b_images, dim=0)  # size [# examples per class, image height, image width]

    number_class_a_examples = torch.tensor(class_a_images.size()[0], device=device)
    print(f"number of used examples from class a: {number_class_a_examples.item()}")
    number_class_b_examples = torch.tensor(class_b_images.size()[0], device=device)
    print(f"number of used examples from class b: {number_class_b_examples.item()}")

    # SET CLASS LABELS TO -1 AND +1
    class_a_labels = -1 * torch.ones(number_class_a_examples, device=device)  # size: [# examples class a]
    class_b_labels = torch.ones(number_class_b_examples, device=device)  # size: [# examples class b]

    # CONCATENATE THE TWO CLASSES
    training_images = torch.cat((class_a_images, class_b_images))  # size: [# examples, image height, image width]
    training_labels = torch.cat((class_a_labels, class_b_labels))  # size: [# examples]
    print(f"total number of examples: {training_labels.size()[0]}")

    # # CHECK PLOT
    # # plot image to check how it compares with normalized and patchified image
    # # uncomment also the plotting of the normalzied and patchified image below
    # example = 18

    # NORMALIZE IMAGES

    # uncomment this to print and learn the mean and std
    # print(f"MNIST mean: {torch.mean(training_images)}")
    # print(f"MNIST std: {torch.std(training_images)}")

    training_images = (training_images - mean) / std

    # # CHECK PLOT
    # # plot image to check if it compares with patchified image
    # # uncomment also the plotting of the patchified image below
    # fig_normalized = plt.figure(figsize=(8, 8))
    # fig_normalized.suptitle('normalized image', fontsize=16)
    # plt.imshow(training_images[example].cpu(), cmap="gray")

    # PATCHIFY (I.E. TOKENIZE) IMAGES
    # add a dummy dimension of size 1 so that images are of the right size for being patchified by unfold
    # (Unfold takes a tensor of size (P, C, *), with P training examples, C color dimension, * whatever
    # - the width and height for us. Here we are adding the dummy color dimension C=1 - i.e. grayscale)
    training_images = torch.unsqueeze(training_images,
                                      dim=1)  # size: [# examples, color=1 , image height, image width]
    training_images = torch.nn.functional.unfold(training_images, kernel_size=patch_linear_size,
                                                 stride=patch_linear_size)
    # size: [# examples, patch_linear_size^2, # tokens = HxW/(patch_linear_size^2)]

    # # CHECK PLOT
    # # plot patches to check if it worked as expected
    # fig_patches = plt.figure(figsize=(8, 8))
    # fig_patches.suptitle('normalized image, tokenized', fontsize=16)
    # grid = ImageGrid(fig_patches, 111, nrows_ncols=(int(28/patch_linear_size), int(28/patch_linear_size)),
    #                  axes_pad=0.1)
    # for i, ax in enumerate(grid):
    #     patch = training_images[example, :, i].view(patch_linear_size, patch_linear_size).cpu().numpy()
    #     ax.imshow(patch, cmap="gray")
    #     ax.axis('off')
    # plt.show()

    # Store some additional info
    dataset_info["number_used_class_a_examples"] = number_class_a_examples.item()
    dataset_info["number_used_class_b_examples"] = number_class_b_examples.item()

    print("\nPREPARING DATASET: END")
    print("\n")

    # unseed everything
    unseed_everything()

    return training_images, training_labels, dataset_info


def prepare_dataset_CIFAR10_binary_regression(dataset_location, dataset_info, train=True,
                                              mean=120.70757293701172, std=64.15007781982422):

    print("\nPREPARING DATASET: START")
    print("\n")
    print("dataset: CIFAR10_binary_regression")

    # here we do everything on RAM and CPU
    device = "cpu"

    # seed everything so we are sure the data is always loaded in the same way
    seed_everything(seed=1)

    # RETRIEVE DATASET INFO
    labels_class_a = dataset_info["labels_class_a"]
    number_labels_class_a = len(labels_class_a)
    labels_class_b = dataset_info["labels_class_b"]
    number_labels_class_b = len(labels_class_b)
    tot_number_labels = number_labels_class_a + number_labels_class_b
    # NOTE: number_examples stores the user specified number of examples. If this is not a multiple of
    # tot_number_labels, this will not represent the actual number of examples (a.k.a. P).
    # Always refer to the training_data size to extract the real number of examples (as the ConvergentSummationHeads
    # class also does)
    number_examples = dataset_info["number_examples"]
    number_examples_per_label = int(number_examples // tot_number_labels)
    patch_linear_size = dataset_info["patch_linear_size"]

    # LOAD DATASET
    training_data = datasets.CIFAR10(
        root=dataset_location,
        train=train,
        download=True,
        transform=ToTensor()
    )
    if 32 % patch_linear_size != 0:
        print(
            f"the chose patch linear size of {patch_linear_size} is not a dividend of 32. Using a patch linear size "
            f"of 8 instead")
        patch_linear_size = 8

    # CONVERT DATA TO TENSORS
    data = torch.tensor(training_data.data, device=device)
    targets = torch.tensor(training_data.targets, device=device)

    # DIVIDE DATA INTO CLASSES A AND B
    # CLASS A
    class_a_images = []
    for label in labels_class_a:
        # exctract images with the given label
        indices = targets == label
        images = data[indices]  # size: [# examples per label, image height, image width, colors]

        # extract only a number "number_examples_per_class" of examples
        images = images[0:number_examples_per_label]
        class_a_images.append(images)
    class_a_images = torch.cat(class_a_images, dim=0)  # size [# examples per class, image height, image width, colors]

    # CLASS B
    class_b_images = []
    for label in labels_class_b:
        # exctract images with the given label
        indices = targets == label
        images = data[indices]  # size: [# examples per label, image height, image width, colors]

        # extract only a number "number_examples_per_class" of examples
        images = images[0:number_examples_per_label]
        class_b_images.append(images)
    class_b_images = torch.cat(class_b_images, dim=0)  # size [# examples per class, image height, image width, colors]

    number_class_a_examples = torch.tensor(class_a_images.size()[0], device=device)
    print(f"number of used examples from class a: {number_class_a_examples.item()}")
    number_class_b_examples = torch.tensor(class_b_images.size()[0], device=device)
    print(f"number of used examples from class b: {number_class_b_examples.item()}")

    # SET CLASS LABELS TO -1 AND +1
    class_a_labels = -1 * torch.ones(number_class_a_examples, device=device)  # size: [# examples class a]
    class_b_labels = torch.ones(number_class_b_examples, device=device)  # size: [# examples class b]

    # CONCATENATE THE TWO CLASSES
    training_images = torch.cat(
        (class_a_images, class_b_images))  # size: [# examples, image height, image width, colors]
    training_labels = torch.cat((class_a_labels, class_b_labels))  # size: [# examples]
    print(f"total number of examples: {training_labels.size()[0]}")

    # CONVERT IMAGES TO FLOAT (CURRENTLY ARE INT) and also to deafault float type
    training_images = training_images.float()
    training_images = training_images.type(torch.get_default_dtype()).to(device)

    # # # CHECK PLOT
    # # # plot image to check how it compares with normalized and patchified image
    # # # uncomment also the plotting of the normalzied and patchified image below
    # example = 3
    # fig_unnormalized = plt.figure(figsize=(8, 8))
    # fig_unnormalized.suptitle('unnormalized image', fontsize=16)
    # example_image = training_images[example].clone()
    # example_image -= torch.min(example_image)
    # example_image /= torch.max(example_image)
    # plt.imshow(example_image.cpu())

    # NORMALIZE IMAGES

    # # uncomment this to print and learn the mean and std
    # print(f"CIFAR10 mean: {torch.mean(training_images)}")
    # print(f"CIFAR10 std: {torch.std(training_images)}")

    training_images = (training_images - mean) / std

    # # CHECK PLOT
    # # plot image to check if it compares with patchified image
    # # uncomment also the plotting of the patchified image below
    # fig_normalized = plt.figure(figsize=(8, 8))
    # fig_normalized.suptitle('normalized image', fontsize=16)
    # example_image = training_images[example].clone()
    # example_image -= torch.min(example_image)
    # example_image /= torch.max(example_image)
    # plt.imshow(example_image.cpu())

    # PATCHIFY (I.E. TOKENIZE) IMAGES
    # reshape training_images from (P, H, W, C) to (P, C, H, W)
    # (Unfold takes a tensor of size (P, C, *), with P training examples, C color dimension, * whatever
    # - the width and height for us. Here we are adding the dummy color dimension C=1 - i.e. grayscale)
    training_images = torch.permute(training_images,
                                    (0, 3, 1, 2))  # size: [# examples, colors , image height, image width]
    training_images = torch.nn.functional.unfold(training_images, kernel_size=patch_linear_size,
                                                 stride=patch_linear_size)
    # size: [# examples, colors*patch_linear_size^2, # tokens = HxW/(patch_linear_size^2)]

    # # CHECK PLOT
    # # plot patches to check if it worked as expected
    # fig_patches = plt.figure(figsize=(8, 8))
    # fig_patches.suptitle('normalized image, tokenized', fontsize=16)
    # example_image = training_images[example].clone()  # size [colors*patch_linear_size^2, # tokens]
    # print(example_image.size())
    # example_image -= torch.min(example_image)
    # example_image /= torch.max(example_image)
    # grid = ImageGrid(fig_patches, 111, nrows_ncols=(int(32/patch_linear_size), int(32/patch_linear_size)),
    #                  axes_pad=0.1)
    # for i, ax in enumerate(grid):
    #     patch = example_image[:, i].view(3, patch_linear_size, patch_linear_size)
    #     patch = torch.permute(patch, (1, 2, 0))
    #     patch = patch.cpu().numpy()
    #     ax.imshow(patch)
    #     ax.axis('off')
    # plt.show()

    # Store some additional info
    dataset_info["number_used_class_a_examples"] = number_class_a_examples.item()
    dataset_info["number_used_class_b_examples"] = number_class_b_examples.item()

    print("\nPREPARING DATASET: END")
    print("\n")

    # unseed everything
    unseed_everything()

    return training_images, training_labels, dataset_info


def print_dataset_info(dataset_info):
    print(f"dataset: {dataset_info['dataset']}")
    if dataset_info["dataset"] == "MNIST_binary_regression" or dataset_info["dataset"] == "CIFAR10_binary_regression":
        print(f"labels_class_a: {dataset_info['labels_class_a']}")
        print(f"labels_class_b: {dataset_info['labels_class_b']}")
        print(f"patch_linear_size: {dataset_info['patch_linear_size']}")


# MODEL

class ConvergentSummationHeads(torch.nn.Module):
    """
    A NOTE ON TEMPERATURE: We used the following convention
    In general, Temperature is never added to the renormalized kernel in functions that compute or take as an input
    the renormalized kernel.
    the renormalized kernel is in general meant to be the bare kernel (i.e. without addition of temperature).
    Temperature is added manually to renormalized_kernel only at the very last moment.
    Specifically, functions that add temperature to the renormalized kernel are:
    compute_energy_action, and compute_predictor_statistics.
    """
    def __init__(self, numbers_heads, model_widths, number_attention_layers, input_width, variances,
                 token_readout_style="average_pooling", attention_nonlinearity="softmax", temperature=0.0):
        """
        Constructor.
        Order parameters are initialized to GP limit.
        Attention weights are initialized with style w_random.

        Parameters
        ----------
        numbers_heads: Iterable[int]
            list of one int specifying the number of heads of the attention layer (i.e. all layers have the same numbers
            of heads)
            OR
            list containing the number of heads in the model, from attention layer 1 to layer L.
            size: [L]
            where L is the number of attention layers

        variances: Iterable[float]
            list of one float specifying the variance of all learnable weights
            OR
            list of variances of the model's learnable weights, in order: linear perceptron, attention layers, readout
            size: [1 + L + 1]
            where L is the number of attention layers

        model_widths: Iterable[int]
            list of one int specifying the width of each hidden layer (i.e. all hidden layers have the same width)
            OR
            list containing the widths at each layer. in order: attention layers, readout (i.e. N1, N2, ..., NL, Na)
            size: [L + 1]
            where L is the number of attention layers

        number_attention_layers: int
            number of attention layers

        input_width: int
            width of the input

        token_readout_style: string
            Default is "average_pooling", which averages over all output tokens.
            Other options:
            "first_token", which reads out from the first token only.
            "last_token", which reads out from the last token only.

        attention_nonlinearity: string
            Default is "softmax".
            Other options:
            "hardmax", takes only the maximum value with probability 1.
        """
        super().__init__()

        print("\n")
        print("MODEL INITIALIZATION: START")
        print("\n")

        # initialize scalar parameters
        self.number_attention_layers = number_attention_layers
        self.input_width = input_width
        self.token_readout_style = token_readout_style
        self.attention_nonlinearity = attention_nonlinearity
        self.temperature = temperature

        # <editor-fold desc="Initialize lists of parameters">
        # The if below (and similarly the ones that follow) is to do the following:
        # if) a list of number_heads is specified (and is of the correct length:
        # number_attention_layers), then use that list.
        # else) if just one number is specified (or the list is of the incorrect length), use that number of heads
        # (or the first element in the list) for all attention layers

        # initialize self.numbers_heads
        # size: [L]
        # L: number_attention_layers
        if len(numbers_heads) == number_attention_layers:
            self.numbers_heads = numbers_heads
        else:
            n_heads = numbers_heads[0]
            self.numbers_heads = []
            for l in range(number_attention_layers):
                self.numbers_heads.append(n_heads)

        # initialize self.total_head_sizes
        # size [L]
        # this contains the total linear size for each order parameter, in the order UL, U(L-1), ..., U(L-l), ..., U1
        # with linear size at L-l given by HL * H(L-1) * ... * H(L-l)
        self.total_head_sizes = []
        tot_n_heads = 1
        for n_heads in list(reversed(self.numbers_heads)):
            tot_n_heads *= n_heads
            self.total_head_sizes.append(tot_n_heads)

        # initialize self.model_widths
        # size: [L + 1]
        # L: number_attention_layers
        if (len(model_widths) - 1) == number_attention_layers:
            self.model_widths = model_widths
        else:
            width = model_widths[0]
            self.model_widths = []
            for l in range(number_attention_layers + 1):
                self.model_widths.append(width)
        # determine the max width of the model (useful for normalizing the cost function)
        self.max_model_width = np.max(self.model_widths)

        # initialize self.variances
        # size: [L + 2]
        # L: number_attention_layers
        if (len(variances) - 2) == number_attention_layers:
            self.variances = variances
        else:
            var = variances[0]
            self.variances = []
            for l in range(number_attention_layers + 2):
                self.variances.append(var)
        # </editor-fold>
        # self.numbers_heads: list of size [L], contains [H1, ..., HL]
        # self.model_widths: list of size [L + 1] contains [N1, ..., NL, Na]
        # self.variances: list of size [L + 2] contains [s0, s1, ..., sL, sa]

        # <editor-fold desc="initialize the attention weights (default is w_random)">
        # the parameters are set to None, and then filled by the initialization function
        # NOTE: here we either have w_attention_weights, or q/k_attention_weights. The weights initializer methods must
        # ensure that either of the two is set to None, when initializing the other.
        # Below, for example, w_attention_weights are initialized with style w_random, while q/k_attention_weights are
        # set to None
        self.attention_weights_style = None
        # possible styles so far:
        # w_random, qk_random
        self.random_attention_weights_seed = None
        self.w_attention_weights = None
        self.q_attention_weights = None
        self.k_attention_weights = None
        self.qk_internal_dimensions = None  # list from attention layer 1 to layer L
        # call the function initializing the w_attention_weights with "w_random" style
        self.set_w_random(random_attention_weights_seed=1)
        # </editor-fold>
        # self.w_attention_weights: list of length L, with L the number of attention layers.
        # contains [W1, W2, ..., WL]
        # each weight in the list is a torch.tensor of size [number_heads at that layer, input_width, input_width]
        # normalization: 1/model_width
        # self.q/k_attention_weights: list of length L, with L the number of attention layers.
        # contains [Q/K1, Q/K2, ..., Q/KL]
        # each weight in the list is a torch.tensor of size [number_heads at layer, qk_internal_dimension, input_width]
        # normalization: 1/sqrt(model_width)

        # initialize the order parameters (to default: GP limit)
        self.current_scalar_order_parameter = self.variances[-1]*torch.ones(1)
        # NOTE: self.current_scalar_order_parameter, a.k.a. Ua, is scalar and is not a learnable Parameter. Rather,
        # we always compute it through the explicit solution of the saddle point. When one of such computations is
        # performed, we update the current value of the scalar order parameter here, using clone().detach()
        # it is initialized here to the GP limit value for elegance (better than having None), but it has no particular
        # meaning, since this parameter is determined as a function of the other order parameters and the training data.
        self.order_parameters = None
        self.set_order_parameters_gp()  # initialized to None, then filled by the initializer function below
        # self.order parameters: list of length L, with L the number of attention layers.
        # contains [UL, U(L-1), ..., U1]
        # The order parameters U(L-l) are square matrices of linear size HL*H(L-1)*...*H(L-l) for l=0,...,L-1
        # To unpack the indices into a tensor, the indices are packed as (ref. self.load() function):
        # h1 h2 ... hL -> (h1 h2 ... hL)

        # define variables filled by self.load()
        self.dataset_info = None
        self.number_training_examples = None
        self.number_tokens = None
        self.attentioned_input = None
        # size of self.attentioned_input: [number_examples, input_width, total_head_size=H1*H2*...*HL]
        # normalization: sqrt(variance_0)/sqrt(input_width * total_head_size)

        # define variables filled by self.sample_bayesian_posterior
        self.posterior_samples = None  # this will be a dictionary containing the posterior samples. The dictionary keys
        # correspond to the name of the network weights:
        # linear perceptron weights: "V_0"
        # attention layer weights: "V_1", "V_2", ..., "V_L" with L the total number of attention layers
        # readout weights: "a". Note that a has shape [number_outputs=1, Na]
        # each of these keys is a numpy array of size [number_samples, ...size_of_weights...]
        self.posterior_sampling_info = {
            "number_runs": 0,  # the number of different times the posterior has been sampled, and the results appended
            "number_chains": [],
            "number_samples_per_chain": [],
            "number_warmups": [],
            "tot_number_samples": [],
            "divergences": [],  # list of list, for each run, it is a list of the divergences at different chains
            "BFMIs": [],  # list of list, as above
            "avg_acceptance_probabilities": [],  # list of list, as above
            "seeds": []
        }
        self.min_temperature = 1.0e-05  # this is the minimum temperature allowed for sampling the posterior.
        # If the model is at a lower or zero temperature, this temperature will be used instead.
        # the min_temperature can be changed using the method self.set_minimal_temperature_posterior_sampling

        # define optional variables to load/store
        self.training_data = None  # loaded by self.store_training_data()
        self.training_labels = None  # loaded by self.store_training_labels()
        self.heads_style_info = None
        # a list of lists of the form [heads_layer_1, heads_layer_2, etc...]
        # each list, e.g. heads_layer_l contains strings describing the head style of each head in layer l

        # print all the initialized info
        self.print_architecture()
        print("\n")
        print("MODEL INITIALIZATION: END")
        print("\n")

    def return_posterior_sampling_filename(self):
        # total number of chains
        number_chains = np.sum(self.posterior_sampling_info["number_chains"])
        # min number samples per chain
        number_samples = np.min(self.posterior_sampling_info["number_samples_per_chain"])
        # min number warmups
        number_warmups = np.min(self.posterior_sampling_info["number_warmups"])
        # seeds
        seeds = self.posterior_sampling_info["seeds"]

        # create string of seeds
        string_list = [str(element) for element in seeds]
        delimiter = "_"
        seeds_list = delimiter.join(string_list)

        filename = "seeds" + seeds_list + f"_Nw{number_warmups}_Ns{number_samples}_Nc{number_chains}"

        return filename

    def print_architecture(self):
        print(f"number_attention_layers: {self.number_attention_layers}")
        print(f"input_width: {self.input_width}")
        print(f"token readout style: {self.token_readout_style}")
        print(f"numbers_heads (H1, H2, ..., HL): {self.numbers_heads}")
        print(f"total_head_sizes (i.e. linear size of matrices (UL, U(L-1), ..., U1): {self.total_head_sizes}")
        print(f"model_widths (N1, N2, ..., NL, Na): {self.model_widths}")
        print(f"max_model_width: {self.max_model_width}")
        print(f"variances (s0, s1, s2, ..., sL, sa): {self.variances}")
        print(f"attention_weights_stile: {self.attention_weights_style}")
        print(f"qk_internal_dimensions (G1, G2, ..., GL): {self.qk_internal_dimensions}")
        print(f"number_training_examples: {self.number_training_examples}")
        print(f"number_tokens: {self.number_tokens}")
        print(f"attention_nonlinearity: {self.attention_nonlinearity}")
        print(f"temperature: {self.temperature}")

    def store_training_data(self, training_data):
        self.training_data = training_data

    def forget_training_data(self):
        self.training_data = None

    def store_training_labels(self, training_labels):
        self.training_labels = training_labels

    def forget_training_labels(self):
        self.training_labels = None

    def store_heads_style_info(self, heads_style_info):
        self.heads_style_info = heads_style_info

    def forget_heads_style_info(self, heads_style_info):
        self.heads_style_info = None

    def to_device(self, device):
        self.to(device)
        if self.attentioned_input is not None:
            self.attentioned_input = self.attentioned_input.to(device)

    def set_w_random(self, random_attention_weights_seed):
        """
        Initializes the attention weights with w_random style.
        NOTE: sets the mutually exclusive parameters, like e.g. self.q_attention_weights to None

        CODING NOTE: device should be set to cpu when drawing random numbers

        Parameters
        ----------
        random_attention_weights_seed: int
            seed for the random weights initialization
        """
        # ATTENTION: device must be set to cpu when drawing random numbers!
        device = "cpu"
        # seed everything before drawing random weights
        seed_everything(random_attention_weights_seed)
        # produce the random attention weights
        attention_weights = []
        for l in range(self.number_attention_layers):
            n_heads = self.numbers_heads[l]
            att_weights = (torch.randn(n_heads, self.input_width, self.input_width, device=device)
                           / torch.tensor(self.input_width))
            # size [number_heads at layer l, input_width, input_width]
            attention_weights.append(att_weights)
        # unseed everything
        unseed_everything()

        # update the parameters of the random weights initialization
        # NOTE: Either we have w_attention_weights, or q/k_attention_weights. The weights initializer method must
        # ensure that either of the two is set to None, when initializing the other.
        # Here, for example, w_attention_weights are initialized with style w_random, while q/k_attention_weights are
        # set to None
        self.random_attention_weights_seed = random_attention_weights_seed
        self.attention_weights_style = "w_random"
        self.w_attention_weights = attention_weights
        self.q_attention_weights = None
        self.k_attention_weights = None
        self.qk_internal_dimensions = None

    def set_qk(self, query_weights, key_weights):
        # takes as input the list of query/key weights from attention layer 1 to layer L.
        # the weights ore of size [# heads, qk_internal_dimension, input_width]

        # check weights list has correct number of layers
        if len(query_weights) != self.number_attention_layers or len(key_weights) != self.number_attention_layers:
            print("ERROR: number of query/key weights given doe snot correspond to the number of layers in the model")
            sys.exit()

        # extract useful info
        qk_internal_dimensions = []
        for weight in query_weights:
            qk_internal_dimensions.append(weight.size()[1])

        # update the parameters of the random weights initialization
        # NOTE: Either we have w_attention_weights, or q/k_attention_weights. The weights initializer method must
        # ensure that either of the two is set to None, when initializing the other.
        self.random_attention_weights_seed = None
        self.attention_weights_style = "qk_user_defined"
        self.w_attention_weights = None
        self.q_attention_weights = query_weights
        self.k_attention_weights = key_weights
        self.qk_internal_dimensions = qk_internal_dimensions

    def set_w(self, w_weights):
        # takes as input the list of w weights from attention layer 1 to layer L.
        # the weights ore of size [# heads, input_width, input_width]

        # check weights list has correct number of layers
        if len(w_weights) != self.number_attention_layers:
            print("ERROR: number of w-weights given does not correspond to the number of layers in the model")
            sys.exit()

        # update the parameters of the random weights initialization
        # NOTE: Either we have w_attention_weights, or q/k_attention_weights. The weights initializer method must
        # ensure that either of the two is set to None, when initializing the other.
        self.random_attention_weights_seed = None
        self.attention_weights_style = "w_user_defined"
        self.w_attention_weights = w_weights
        self.q_attention_weights = None
        self.k_attention_weights = None
        self.qk_internal_dimensions = None

    def set_qk_random(self, random_attention_weights_seed, qk_internal_dimensions):
        """
        Initializes the attention weights with qk_random style.
        NOTE: sets the mutually exclusive parameters, like e.g. self.w_attention_weights to None

        CODING NOTE: device should be set to cpu when drawing random numbers

        Parameters
        ----------
        random_attention_weights_seed: int
            seed for the random weights initialization

        qk_internal_dimension: Iterable[int]
            the internal features dimension of query and key weights, from layer 1 to layer L
            (these weights have size [# heads, qk_internal_dimension, input_dimension])
        """
        # ATTENTION: device must be set to cpu when drawing random numbers!
        device = "cpu"

        # check if query_key_internal_dimensions has the correct length, otherwise use the first entry for all layers
        if len(qk_internal_dimensions) == self.number_attention_layers:
            self.qk_internal_dimensions = qk_internal_dimensions
        else:
            qk_dim = qk_internal_dimensions[0]
            self.qk_internal_dimensions = []
            for l in range(self.number_attention_layers):
                self.qk_internal_dimensions.append(qk_dim)
        # seed everything before drawing random weights
        seed_everything(random_attention_weights_seed)
        # produce the random attention weights
        q_weights = []
        k_weights = []
        for l in range(self.number_attention_layers):
            qk_internal_dimension = self.qk_internal_dimensions[l]
            n_heads = self.numbers_heads[l]
            q_weig = (torch.randn(n_heads, qk_internal_dimension, self.input_width, device=device)
                             / torch.sqrt(torch.tensor(self.input_width)))
            k_weig = (torch.randn(n_heads, qk_internal_dimension, self.input_width, device=device)
                             / torch.sqrt(torch.tensor(self.input_width)))

            # size [number_heads at layer l, qk_internal_dimension, input_width]
            q_weights.append(q_weig)
            k_weights.append(k_weig)
        # unseed everything
        unseed_everything()

        # update the parameters of the random weights initialization
        # NOTE: Either we have w_attention_weights, or q/k_attention_weights. The weights initializer method must
        # ensure that either of the two is set to None, when initializing the other.
        # Here, for example, w_attention_weights are initialized with style w_random, while q/k_attention_weights are
        # set to None
        self.random_attention_weights_seed = random_attention_weights_seed
        self.attention_weights_style = "qk_random"
        self.w_attention_weights = None
        self.q_attention_weights = q_weights
        self.k_attention_weights = k_weights
        # self.qk_internal_dimensions has already been updated above

    def set_order_parameters_gp(self):
        """
        Initializes the order parameters to the GP limit. i.e. U_l = tot_variance_l * Id
        with tot_variance_l = var_a * var_L * var_(L-1) * ... * var_(L-l)

        CODING NOTE: Any initializer of the order parameters should make sure they are symmetric.

        CODING NOTE: Any initializer of the order parameters should make sure they are positive definite! As we do here
        by adding Marchenko-Pastur (as opposed to e.g. just a random Gaussian)
        """

        # NOTE: we need to cycle through the layers from top to bottom!
        order_parameters_list = []
        tot_size = 1  # this counts the product of HL * H(L-1) * ... * HL-l, i.e. the linear size of the order
        tot_variance = self.variances[-1]  # this counts the product var_a * var_L * var_(L-1) * ... * var_(L-l)
        # parameter at L-l
        for l in range(self.number_attention_layers):
            variance = self.variances[-l-2]  # we start from the second entry from the top, i.e. sigma_L
            n_heads = self.numbers_heads[-l-1]  # we start from the first entry from the top, i.e. H_L
            tot_size *= n_heads
            tot_variance *= variance
            order_parameter = torch.nn.Parameter(tot_variance*torch.eye(tot_size))
            order_parameters_list.append(order_parameter)
        # transform the list of Parameters into an iterable that nn.Module can properly see as a parameters list.
        self.order_parameters = torch.nn.ParameterList(order_parameters_list)

    def set_order_parameters_gp_perturbed(self, perturbation_strength, scale=1.0, seed=None):
        """
        Initializes the order parameters to the GP limit + a random perturbation.
        GP limit is: U_l = tot_variance_l * Id
        with tot_variance_l = var_a * var_L * var_(L-1) * ... * var_(L-l)
        The random perturbation P is Marchenko-Pastur, with m=n=linear_size(U_l) and the order parameter will be:
        U_l = variance_l *  (Id + perturbation_strength*P)

        CODING NOTE: device should be set to cpu when drawing random numbers

        CODING NOTE: Any initializer of the order parameters should make sure they are symmetric.

        CODING NOTE: Any initializer of the order parameters should make sure they are positive definite! As we do here
        by adding Marchenko-Pastur (as opposed to e.g. just a random Gaussian)

        Parameters
        ----------
        perturbation_strength: float
            strength of the random perturbation

        """
        # ATTENTION: device must be set to cpu when drawing random numbers!
        device = "cpu"

        # seed everything, if required
        if seed is not None:
            seed_everything(seed=seed)

        # NOTE: we need to cycle through the layers from top to bottom!
        order_parameters_list = []
        tot_size = 1  # this counts the product of HL * H(L-1) * ... * HL-l, i.e. the linear size of the order
        tot_variance = self.variances[-1]  # this counts the product var_a * var_L * var_(L-1) * ... * var_(L-l)

        # parameter at L-l
        for l in range(self.number_attention_layers):
            variance = self.variances[-l-2]  # we start from the second entry from the top, i.e. sigma_L
            n_heads = self.numbers_heads[-l-1]  # we start from the first entry from the top, i.e. H_L
            tot_size *= n_heads
            tot_variance *= variance
            gaussian_matrix = torch.randn(tot_size, tot_size, device=device)
            marchenko_pastur = (torch.matmul(gaussian_matrix, torch.transpose(gaussian_matrix, 0, 1))
                                / torch.tensor(tot_size))
            order_parameter = torch.nn.Parameter(tot_variance*(torch.eye(tot_size) +
                                                               perturbation_strength*marchenko_pastur))

            # scale the order parameter by the user-defined value
            order_parameter = scale*order_parameter

            order_parameters_list.append(order_parameter)

        # transform the list of Parameters into an iterable that nn.Module can properly see as a parameters list.
        self.order_parameters = torch.nn.ParameterList(order_parameters_list)

        # unseed everything
        unseed_everything()

    def set_order_parameters_good_vs_bad_heads(self, number_good_heads=1, scale=1.0, reduce_factor_bad_heads=0.1):
        """
        CODING NOTE: device should be set to cpu when drawing random numbers

        CODING NOTE: Any initializer of the order parameters should make sure they are symmetric.

        CODING NOTE: Any initializer of the order parameters should make sure they are positive definite!
        """
        # ATTENTION: this only works if the good heads are those appearing first

        # NOTE: we need to cycle through the layers from top to bottom!
        order_parameters_list = []
        tot_size = 1  # this counts the product of HL * H(L-1) * ... * HL-l, i.e. the linear size of the order

        # parameter at L-l
        for l in range(self.number_attention_layers):
            n_heads = self.numbers_heads[-l-1]  # we start from the first entry from the top, i.e. H_L
            tot_size *= n_heads

            # intialize the order parameter to the value for the bad heads
            order_param = torch.eye(tot_size)*reduce_factor_bad_heads*scale

            # set the value for the good heads
            for h in range(number_good_heads):
                order_param[h,h] = scale

            order_parameter = torch.nn.Parameter(order_param)

            order_parameters_list.append(order_parameter)

        # transform the list of Parameters into an iterable that nn.Module can properly see as a parameters list.
        self.order_parameters = torch.nn.ParameterList(order_parameters_list)

    def compute_attentioned_input(self, input):
        # compute the attention matrix of the last layer
        attention_matrix = self.compute_attention_matrix(input=input,
                                                         layer=self.number_attention_layers)
        # size [number_examples, number_heads, number_tokens, number_tokens]
        # evaluate the last token at t* (meaning of t* can be average pooling or first token prediction)
        attention_vector = self.apply_token_readout_style(attention_matrix)
        # size [number_examples, number_heads, number_tokens]

        # iterate through the remaining layers, to compute the product of attention matrices
        for l in range(self.number_attention_layers - 1):
            # the (number_attention_layers - 1) above is because we already did the last layer outside the loop
            # NOTE: here we iterate through the layers backwards, starting from the final layers, down to the first
            attention_matrix = self.compute_attention_matrix(input=input,
                                                             layer=(self.number_attention_layers - l - 1))
            # compute the dot product of the previous attention vector with the new attention matrix
            attention_vector = einsum(attention_matrix, attention_vector, "b h t s, b H s -> b h H t")
            # collapse all head indices into one
            # NOTE: the way the collapsing is done is, in the end of the loop, the following
            # h1 h2 ... hL -> (h1 h2 ... hL)
            attention_vector = rearrange(attention_vector, "b h H t -> b (h H) t")

        # obtain the attentioned_input by attentioning it with the attention_vector
        attentioned_input = einsum(input, attention_vector, "b i s, b H s -> b i H")
        # compute H = H1*H2*...*HL
        total_head_size = torch.tensor(attentioned_input.size()[-1])
        std = torch.sqrt(torch.tensor(self.variances[0]))
        attentioned_input = std * attentioned_input / torch.sqrt(torch.tensor(self.input_width) * total_head_size)
        # size [number_examples, input_width, H1*H2*...*HL]
        # normalization: sqrt(variance_0)/sqrt(input_width * total_head_size)

        return attentioned_input

    def load(self, training_input, dataset_info):
        """
        Loads the training data in the model.
        Specifically, it constructs self.attentioned_input from the training data.

        Parameters
        ----------
        training_input: torch.Tensor
            training data, of size [# examples, input_width, # tokens]

        dataset_info: dictionary
            dictionary containing any relevant information for generating the training_input.
            This is fundamental to store! Since we do not store the training_input itself nor any derived quantity
            (like e.g. self.attentioned_input) when we store the model, we need to recall exactly what was the training
            set it was trained on, in oder to reinitialize the model, e.g. to make predictions.
        """
        # check that the size of the training input is consistent
        if self.input_width != training_input.size()[1]:
            print("ERROR: the width of the training input provided does not match with the input width of the model")
            sys.exit()

        # extract information to store into the model parameters
        self.dataset_info = dataset_info
        self.number_training_examples = training_input.size()[0]
        self.number_tokens = training_input.size()[2]

        # compute the attentioned input

        self.attentioned_input = self.compute_attentioned_input(training_input)
        # size [number_examples, input_width, H1*H2*...*HL]
        # normalization: sqrt(variance_0)/sqrt(input_width * total_head_size)

    def unload(self):
        """
        Unloads the training data in the model.
        Specifically, it unloads self.attentioned_input.
        This function should be called before storing the model, since we do not want to keep heavy information
        on the training input
        """
        self.attentioned_input = None

    def unload_before_checkpoint(self):
        attentioned_input = self.attentioned_input
        self.unload()
        return attentioned_input

    def load_after_checkpoint(self, attentioned_input):
        self.attentioned_input = attentioned_input

    def apply_token_readout_style(self, attention_matrix):
        """
        Applies the token readout style to the attention matrix.
        Specifically, if self.readout_style is
        a) "average_pooling": it averages over the query token
        b) "first_token": it evaluates at the first query token
        C) "last_token": it evaluates at the last query token

        Parameters
        ----------
        attention_matrix: torch.Tensor
            tensor of size [number_examples, number_heads, number_tokens, number_tokens]
        """
        if self.token_readout_style == "average_pooling":
            # average over the query token
            attention_vector = reduce(attention_matrix, "b h s t -> b h s", "mean")

        elif self.token_readout_style == "first_token":
            # look only at the first of the query tokens
            # attention matrix is of size [number_examples, number_heads, number_tokens, number_tokens]
            attention_vector = attention_matrix[:, :, :, 0]

        elif self.token_readout_style == "last_token":
            # look only at the last of the query tokens
            # attention matrix is of size [number_examples, number_heads, number_tokens, number_tokens]
            attention_vector = attention_matrix[:, :, :, -1]

        else:
            print("\nWARNING: the specified token_readout_option is not valid. Using average_pooling instead\n")
            # average over the query token
            attention_vector = reduce(attention_matrix, "b h s t -> b h s", "mean")

        return attention_vector

    def compute_attention_matrix(self, input, layer):
        """
        Computes the attention matrix on the training input, at the specified layer.
        Uses self.w_attention_weights ot self.q/k_attention_weights, depending on which one is initialized in the model

        Parameters
        ----------
        input: torch.Tensor
            training/test data, of size [# examples, input_width, # tokens]

        layer: int
            attention layer for which to compute the attention matrix
            NOTE: the layer can be {1, 2, ..., L}, i.e. it is not indexed from 0 to L-1 but from 1 to L
        """

        if self.w_attention_weights is not None:
            return self.compute_attention_matrix_w(input, layer)
        else:
            return self.compute_attention_matrix_qk(input, layer)

    def compute_attention_matrix_w(self, input, layer):
        """
        Computes the attention matrix on the training input, at the specified layer, using self.w_attention_weights
        (as opposed to using self.q/k_attention_weights)

        Parameters
        ----------
        input: torch.Tensor
            training/test data, of size [# examples, input_width, # tokens]

        layer: int
            attention layer for which to compute the attention matrix
            NOTE: the layer can be {1, 2, ..., L}, i.e. it is not indexed from 0 to L-1 but from 1 to L
        """
        # below, layer-1 is used because indices start counting from 0
        pre_attention = einsum(input, self.w_attention_weights[layer - 1], input,
                               "b i s, h i j, b j t -> b h s t")
        # size [number_examples, number_heads, number_tokens, number_tokens]

        if self.attention_nonlinearity == "hardmax":
            hardmax_result = torch.zeros_like(pre_attention)
            max_indices = torch.argmax(pre_attention, dim=2)
            attention_matrix = hardmax_result.scatter_(2, max_indices.unsqueeze(2), 1)
            # attention is of size [# examples, # heads, # tokens, # tokens]
        else:
            if self.attention_nonlinearity != "softmax":
                print("\nWARNING: the required attention nonlinearity " + self.attention_nonlinearity +
                      "is not implemented/valid (at least for attention weights of style w. "
                      "Using softmax instead")
            # softmax is done on the first of the token indices, i.e. that coming from the Key
            attention_matrix = torch.softmax(pre_attention, dim=2)

        return attention_matrix

    def compute_attention_matrix_qk(self, input, layer):
        """
        Computes the attention matrix on the training input, at the specified layer, using self.q/k_attention_weights
        (as opposed to using self.w_attention_weights)

        Parameters
        ----------
        input: torch.Tensor
            training/test data, of size [# examples, input_width, # tokens]

        layer: int
            attention layer for which to compute the attention matrix
            NOTE: the layer can be {1, 2, ..., L}, i.e. it is not indexed from 0 to L-1 but from 1 to L
        """
        # below, layer-1 is used because indices start counting from 0
        qk_internal_dimension = self.qk_internal_dimensions[layer - 1]
        queries = einsum(input, self.q_attention_weights[layer - 1], "b i t, h q i -> b h q t")
        keys = einsum(input, self.k_attention_weights[layer - 1], "b i t, h q i -> b h q t")
        # size [number_examples, number_heads, qk_internal_dimension, number_tokens]
        pre_attention = (einsum(keys, queries, "b h q s, b h q t -> b h s t")
                            / torch.sqrt(torch.tensor(qk_internal_dimension)))
        # size [number_examples, number_heads, number_tokens, number_tokens]
        # softmax is done on the first of the token indices, i.e. that coming from the Key

        if self.attention_nonlinearity != "softmax":
            print("\nWARNING: the required attention nonlinearity " + self.attention_nonlinearity +
                  "is not implemented/valid (at least for attention weights of style qk. "
                  "Using softmax instead")
        attention_matrix = torch.softmax(pre_attention, dim=2)

        return attention_matrix

    def return_attention_matrices(self, input, numpy=True):
        # NOTE: the attention matrix of the last layer is actually an attention vector,
        # i.e. we apply the token readout style
        # compute the attention matrices
        attention_matrices = []
        for layer in range(self.number_attention_layers):
            # we use layer+1 because attention self.compute_attention_matrix indices attention layers starting from 1
            attn_matrix = self.compute_attention_matrix(input=input, layer=(layer+1))
            # apply the token readout style, if this is the matrix of the last layer
            if (layer + 1) == self.number_attention_layers:
                attention_vector = self.apply_token_readout_style(attn_matrix)
                if numpy:
                    attention_matrices.append(attention_vector.detach().clone().cpu().numpy())
                else:
                    attention_matrices.append(attention_vector)
            else:
                if numpy:
                    attention_matrices.append(attn_matrix.detach().clone().cpu().numpy())
                else:
                    attention_matrices.append(attn_matrix)

        return attention_matrices

    def compute_loss_action(self, labels, return_energy_entropy=False):
        # CODING NOTE: the symmetrized order parameter must be generated at the beginning of the computation, and used
        # throughout. The self.order_parameters should not be used directly!
        # Recall: self.order_parameters is upper triangular (i.e. contains only the independent parameters)
        # symmetrized order parameter symmetrizes these upper triangular parameters, so that can be plugged into the
        # equations.

        symmetrized_order_parameters = self.compute_symmetrized_order_parameters()
        # NOTE: symmetrized_order_parameters are produced in the same order as self.order_parameters, i.e.
        # [UL, U(L-1), ..., U1]

        entropy = self.compute_entropy_action(symmetrized_order_parameters)
        # the above function updates self.current_scalar_order_parameter with clone().detach()

        energy = self.compute_energy_action(symmetrized_order_parameters[-1], labels)

        # we normalize by the max width in the model
        energy /= self.max_model_width
        entropy /= self.max_model_width

        if return_energy_entropy:
            return energy, entropy
        else:
            cost_function = entropy + energy
            return cost_function

    def compute_energy_action(self, symmetrized_order_parameter_largest, labels):
        # NOTE: symmetrized_order_parameter_largest is U_1, i.e. the one that renormalizes the Kernel

        renormalized_kernel = self.compute_renormalized_kernel(symmetrized_order_parameter_largest)

        # add temperature to the kernel
        renormalized_kernel += torch.tensor(self.temperature)*torch.eye(renormalized_kernel.size()[0],
                                                                        device=renormalized_kernel.device)

        mean_squared_readout_nonnormalized = self.compute_mean_squared_readout_nonnormalized(renormalized_kernel,
                                                                                             labels)

        logdet_kernel = torch.logdet(renormalized_kernel)

        energy = logdet_kernel + mean_squared_readout_nonnormalized

        # USEFUL FOR DEBUGGING: set energy to zero
        # energy = 0.0 * energy
        # END USEFUL FOR DEBUGGING

        return energy

    def compute_entropy_action(self, symmetrized_order_parameters):
        # NOTE: inside this function self.current_scalar_order_parameter is updated with .detach().clone()

        # symmetrized_order_parameters are in the same order as self.order_parameters, i.e. [UL, U(L-1), ..., U1]

        # compute the scalar order parameter, as a function of U_L
        scalar_order_parameter = self.compute_scalar_order_parameter(symmetrized_order_parameters[0])
        # the above function updates self.current_scalar_order_parameter with .detach().clone()

        # We first compute the entropy for the scalar (U_a) and first order parameter (U_L) outside the loop
        var_readout = self.variances[-1]  # variance of the readout: sigma^2_a
        var_last_layer = self.variances[-2]  # variance of the last attention layer: sigma^2_L
        width_readout = self.model_widths[-1]  # width of the readout layer: N_a
        width_last_layer = self.model_widths[-2]  # width of the last attention layer: N_L
        n_heads_last_layer = self.numbers_heads[-1]  # number of heads of the last attention layer: H_L
        # compute entropy of scalar order parameter (U_a)
        logdet_current = torch.log(scalar_order_parameter)
        entropy_scalar = width_readout * (scalar_order_parameter / var_readout - logdet_current)

        # compute entropy of first order parameter (U_L)
        logdet_previous = logdet_current.clone()
        logdet_current = torch.logdet(symmetrized_order_parameters[0])
        entropy_last_layer = (width_last_layer * (
                              torch.trace(symmetrized_order_parameters[0])/(scalar_order_parameter*var_last_layer)
                              - logdet_current
                              + n_heads_last_layer * logdet_previous))

        tot_entropy = entropy_scalar + entropy_last_layer

        # Loop over the remaining layers
        for l in range(self.number_attention_layers - 1):
            # the -1 is because we already computed the last layer (U_L) outside the loop

            # retrieve all necessary quantities
            var_current = self.variances[-2 - 1 - l]  # we start with var_(L-1)
            width_current = self.model_widths[-2 - 1 - l]  # we start with N_(L-1)
            n_heads_current = self.numbers_heads[-2 - l]  # we start with H_(L-1)
            order_param_previous = symmetrized_order_parameters[l]  # we start with U_L
            order_param_current = symmetrized_order_parameters[1 + l]  # we start with U_(L-1)
            logdet_previous = logdet_current.clone()
            logdet_current = torch.logdet(order_param_current)

            # unpack the order parameter, making explicit the head indices of the current layer
            order_param_current_unpacked = rearrange(order_param_current, "(h1 H1) (h2 H2) -> h1 H1 h2 H2",
                                                     h1=n_heads_current, h2=n_heads_current)

            # trace over the indices of the current layer
            order_param_current_partially_traced = einsum(order_param_current_unpacked, "h H1 h H2 -> H1 H2")

            # compute (U_L)^-1 . u_(L-1) (where u_(L-1) is the partially traced order param)
            order_params_product = torch.linalg.solve(order_param_previous, order_param_current_partially_traced)

            # compute the entropy
            entropy_current = width_current * (torch.trace(order_params_product)/var_current
                                               - logdet_current
                                               + n_heads_current * logdet_previous)

            tot_entropy += entropy_current

        return tot_entropy

    def compute_symmetrized_order_parameters(self):
        # NOTE: symmetrized_order_parameters are produced in the same order as self.order_parameters, i.e.
        # [UL, U(L-1), ..., U1]
        symmetrized_order_parameters = []
        for l in range(self.number_attention_layers):
            order_param = self.order_parameters[l].clone()  # clone is probably unnecessary, but let's do it

            # symmetrize:
            symm_order_param = (order_param + torch.transpose(order_param, 0, 1)) / 2

            symmetrized_order_parameters.append(symm_order_param)

        return symmetrized_order_parameters

    def compute_symmetrized_order_parameter_largest(self):
        # NOTE: symmetrized_order_parameters are produced in the same order as self.order_parameters, i.e.
        # [UL, U(L-1), ..., U1]

        order_param = self.order_parameters[-1].clone()  # clone is probably unnecessary, but let's do it

        # symmetrize:
        symm_order_param = (order_param + torch.transpose(order_param, 0, 1)) / 2

        return symm_order_param

    def compute_packed_order_parameters(self):
        symmetrized_order_parameters = self.compute_symmetrized_order_parameters()

        # for each order parameter, collapse it from a matrix (the upper triagular only)
        # to a vector and append it to the list
        # NOTE: here we take the upper triangular part only, because these are the only independent parameters.
        # If we didn't do this, then the Hessian could have negative eigenvalues, which are however irrelevant,
        # as they correspond to directions that are asymmetric, and thus forbidden by the symmetry of the order
        # parameter
        order_params_list = []
        for order_param in symmetrized_order_parameters:
            order_param_packed = order_param.clone()[torch.triu(torch.ones(order_param.size())) == 1]
            order_params_list.append(order_param_packed)

        # stack the list of vectorial order parameters into a single vector
        packed_order_parameters = torch.cat(order_params_list)

        return packed_order_parameters

    def compute_loss_action_for_hessian(self, packed_order_parameters):
        # unpack the order parameters
        symmetrized_order_parameters = self.unpack_order_parameters(packed_order_parameters)

        entropy = self.compute_entropy_action(symmetrized_order_parameters)
        # the above function updates self.current_scalar_order_parameter with clone().detach()

        energy = self.compute_energy_action(symmetrized_order_parameters[-1], self.training_labels)

        # we normalize by the max width in the model
        energy /= self.max_model_width
        entropy /= self.max_model_width

        cost_function = entropy + energy
        return cost_function

    def unpack_order_parameters(self, packed_order_parameters):
        # this does the inverse operation of self.compute_packed_order_parameters()

        # split the big vector of order parameters into vectors, each containing one order parameter
        # list of H*(H+1)/2, which is the size of each vectorized order parameter to unpack
        # (i.e. number of elements of the upper triangular part of the order param in matrix form)
        total_sizes = [int(H*(H+1)/2) for H in self.total_head_sizes]
        # split does the inverse of cat, which we used in self.compute_packed_order_parameters()
        splitted_order_parameters = torch.split(packed_order_parameters.clone(), total_sizes)

        # restore each vectorized order parameter into matrix form
        # i.e. do the opposite of the upper triangular extraction done in self.compute_packed_order_parameters()
        unpacked_order_parameters = []
        for l, order_param_vector in enumerate(splitted_order_parameters):
            tot_head_size = self.total_head_sizes[l]
            order_param_matrix = torch.zeros(tot_head_size, tot_head_size,
                                             device=order_param_vector.device)
            order_param_matrix[torch.triu(torch.ones(order_param_matrix.size())) == 1] = order_param_vector.clone()
            unpacked_order_parameters.append(order_param_matrix)

        # symmetrize the order parameters
        # NOTE: here we symmetrize with a slightly different method than in the function
        # self.compute_symmetrized_order_parameters(), because we start with order parameters which onyl have the
        # upper triangular part
        symmetrized_order_parameters = []
        for l in range(self.number_attention_layers):
            order_param = unpacked_order_parameters[l].clone()  # clone is probably unnecessary, but let's do it

            symm_order_param = (torch.diag_embed(torch.diagonal(order_param))
                                + torch.triu(order_param, diagonal=1)
                                + torch.transpose(torch.triu(order_param, diagonal=1), 0, 1))

            symmetrized_order_parameters.append(symm_order_param)

        return symmetrized_order_parameters

    def compute_renormalized_kernel(self, symmetrized_order_parameter_largest):
        # self.attentioned_input is of size: [number_examples, input_width, total_head_size=H1*H2*...*HL]
        # and normalization: sqrt(variance_0)/sqrt(input_width * total_head_size)
        # effective_order_parameter_last_layer is of size [total_head_size,total_head_size]

        renormalized_attentioned_input = self.compute_renormalized_attentioned_input(
            symmetrized_order_parameter_largest)

        renormalized_kernel = einsum(self.attentioned_input, renormalized_attentioned_input,
                                     "b1 i H, b2 i H -> b1 b2")

        return renormalized_kernel

    def test_gp_kernel_invertibility(self, plot=False, with_temperature=False):

        gp_kernel = self.return_gp_kernel().detach()

        if with_temperature:
            gp_kernel += torch.tensor(self.temperature)*torch.eye(gp_kernel.size()[0])

        eigvals = torch.linalg.eigvalsh(gp_kernel)
        print("Eigenvalues")
        print(eigvals)
        min_eigval = torch.min(eigvals)
        max_eigval = torch.max(eigvals)
        print("\n")
        print("TEST INVERTIBILITY OF GP KERNEL: START")
        print("\n")
        print(f"Test performed including temperature? {with_temperature}")
        print(f"minimum eigenvalue: {min_eigval}")
        print(f"maximum eigenvalue: {max_eigval}")
        number_negative_eigenvalues = torch.argwhere(eigvals <= 0).size()[0]
        print(f"number of negative (or zero) eigenvalues: {number_negative_eigenvalues}")
        if number_negative_eigenvalues > 0:
            print("WARNING: Certainly above capacity")
        else:
            print("No negative (or zero) eigenvalues found")
            print("NOTE: this does not guarantee to be below capacity.\nFor complete certainty, plot the eigenvalues "
                  "and do the double peaked distribution test")
        print("\n")
        print("TEST INVERTIBILITY OF GP KERNEL: END")
        print("\n")
        if plot:
            fig_kernel_invertibility = plt.figure(figsize=(8, 8))
            fig_kernel_invertibility.suptitle("GP kernel eigenvalues", fontsize=16)
            plt.hist(eigvals.cpu(),
                     bins=np.geomspace(start=np.max([min_eigval.cpu(), 10 ** (-10)]), stop=max_eigval.cpu(),
                                       num=max(int(gp_kernel.size()[0] / 30), 20)))
            plt.gca().set_xscale("log")
            plt.savefig('./test_gp_kernel_invertibility.png')
            # plt.show()

    def return_gp_kernel(self):
        # self.attentioned_input is of size: [number_examples, input_width, total_head_size=H1*H2*...*HL]
        # and normalization: sqrt(variance_0)/sqrt(input_width * total_head_size)

        tot_variance = 1
        for l in range(self.number_attention_layers + 1):
            tot_variance *= self.variances[l+1]  # the +1 is because we start from sigma_1, excluding sigma_0

        gp_kernel = (einsum(self.attentioned_input, self.attentioned_input, "b1 i H, b2 i H -> b1 b2")
                     * torch.tensor(tot_variance))

        return gp_kernel

    def return_pre_kernels(self):
        # NOTE: ATTENTION! we return the pre_kernels without normalization 1/total_head_size !!!
        # self.attentioned_input is of size: [number_examples, input_width, total_head_size=H1*H2*...*HL]
        # and normalization: sqrt(variance_0)/sqrt(input_width * total_head_size)

        total_head_size = torch.tensor(self.attentioned_input.size()[-1])

        pre_kernels = (einsum(self.attentioned_input, self.attentioned_input, "b1 i H1, b2 i H2 -> H1 H2 b1 b2")
                       * total_head_size)
        # size [tot_number_heads, tot_number_heads, # examples, # examples]

        return pre_kernels

    def plot_pre_kernels(self, style="all", with_temperature=False):

        pre_kernels = self.return_pre_kernels().detach().clone().cpu().numpy()
        total_head_size = np.shape(pre_kernels)[0]

        # <editor-fold desc="create custom labels">
        indices = np.empty(self.numbers_heads, dtype=object)

        # Create a list of range objects based on the H values
        ranges = [range(H) for H in self.numbers_heads]

        # Use itertools.product to generate all combinations
        for combination in product(*ranges):
            # combination is a tuple containing values for h1, h2, ..., hL
            # You can access individual values like this:
            # h1, h2, h3, ..., hL = combination
            string = ""
            for i, index in enumerate(combination):
                if i == 0:  # do not put a "-" if this is the first index
                    string += f"{index}"
                else:
                    string = string + "-" + f"{index}"
            indices[combination] = string

        for l in range(len(self.numbers_heads) - 1):
            pre_arrangement = ""
            post_arrangement = ""
            for H in range(len(self.numbers_heads) - l):
                if H == 0:
                    pre_arrangement = f"h{H}"
                    post_arrangement = f"h{H})"
                elif H == 1:
                    pre_arrangement = f"h{H} " + pre_arrangement
                    post_arrangement = f"(h{H} " + post_arrangement
                else:
                    pre_arrangement = f"h{H} " + pre_arrangement
                    post_arrangement = f"h{H} " + post_arrangement
            indices = rearrange(indices, pre_arrangement + " -> " + post_arrangement)
        # </editor-fold>

        # CREATE HEADS STYLE LABELS, if self.heads_style_info is present
        if self.heads_style_info is not None:
            # infos = np.array(self.heads_style_info)
            head_infos = np.empty(self.numbers_heads, dtype=object)

            # Create a list of range objects based on the H values
            ranges = [range(H) for H in self.numbers_heads]

            # Use itertools.product to generate all combinations
            for combination in product(*ranges):
                # combination is a tuple containing values for h1, h2, ..., hL
                # You can access individual values like this:
                # h1, h2, h3, ..., hL = combination
                string = ""
                for layer, head in enumerate(combination):
                    if layer == 0:  # do not put a "-" if this is an head in the first layer
                        string = string + self.heads_style_info[layer][head]
                    else:
                        string = string + " -> \n" + self.heads_style_info[layer][head]
                head_infos[combination] = string

            for l in range(len(self.numbers_heads) - 1):
                pre_arrangement = ""
                post_arrangement = ""
                for H in range(len(self.numbers_heads) - l):
                    if H == 0:
                        pre_arrangement = f"h{H}"
                        post_arrangement = f"h{H})"
                    elif H == 1:
                        pre_arrangement = f"h{H} " + pre_arrangement
                        post_arrangement = f"(h{H} " + post_arrangement
                    else:
                        pre_arrangement = f"h{H} " + pre_arrangement
                        post_arrangement = f"h{H} " + post_arrangement
                head_infos = rearrange(head_infos, pre_arrangement + " -> " + post_arrangement)

        if style == "all":
            for h1 in range(total_head_size):
                for h2 in range(h1, total_head_size):
                    plt.figure()
                    pre_kernel = pre_kernels[h1, h2, :, :]
                    if with_temperature:
                        pre_kernel += self.temperature * np.eye(np.shape(pre_kernel)[0])
                    plt.imshow(pre_kernel, cmap='viridis')
                    plt.colorbar()  # Add a colorbar to show the scale
                    plt.title(f'pre-kernel {indices[h1]} and {indices[h2]}')
                    if self.heads_style_info is not None:
                        # Adding text below the plot
                        # noinspection PyUnboundLocalVariable
                        subtext = "\n" + head_infos[h1] + "\n AND \n" + head_infos[h2]
                        plt.text(0.5, -0.1, subtext, ha='center', va='top', fontsize=12,
                                 transform=plt.gca().transAxes)
                    plt.tight_layout()
                    # else:
                    #     plt.title(f'pre-kernel {indices[h1]} and {indices[h2]}' +
                    #               "\n" + head_infos[h1] + "\n AND \n" + head_infos[h2])
                        # plt.title("hello")

        elif style == "diagonal":
            for h1 in range(total_head_size):
                plt.figure()
                pre_kernel = pre_kernels[h1, h1, :, :]
                if with_temperature:
                    pre_kernel += self.temperature * np.eye(np.shape(pre_kernel)[0])
                plt.imshow(pre_kernel, cmap='viridis')
                plt.colorbar()  # Add a colorbar to show the scale

                plt.title(f'pre-kernel {indices[h1]} and {indices[h1]}')
                if self.heads_style_info is not None:
                    # Adding text below the plot
                    # noinspection PyUnboundLocalVariable
                    subtext = "\n" + head_infos[h1]
                    plt.text(0.5, -0.1, subtext, ha='center', va='top', fontsize=12,
                             transform=plt.gca().transAxes)
                plt.tight_layout()

    def plot_kernel(self, with_temperature=False):
        gp_kernel = self.return_gp_kernel()
        renormalized_kernel = self.return_renormalized_kernel()

        if with_temperature:
            gp_kernel += torch.tensor(self.temperature)*torch.eye(gp_kernel.size()[0])
            renormalized_kernel += torch.tensor(self.temperature) * torch.eye(renormalized_kernel.size()[0])

        plt.figure()
        plt.imshow(gp_kernel.detach().clone().cpu().numpy(), cmap='viridis')
        plt.colorbar()  # Add a colorbar to show the scale
        plt.title(f'GP kernel')

        plt.figure()
        plt.imshow(renormalized_kernel.detach().clone().cpu().numpy(), cmap='viridis')
        plt.colorbar()  # Add a colorbar to show the scale
        plt.title(f'renormalized kernel')

    def plot_order_parameter(self, order_parameter=None, plot_order_parameter_file_name=None):
        # retrieve order parameter from model, if it is not given by user
        if order_parameter is None:
            order_parameter = self.compute_symmetrized_order_parameter_largest().detach().clone().cpu().numpy()
        plt.figure()
        plt.imshow(order_parameter, cmap='viridis')
        plt.colorbar()  # Add a colorbar to show the scale
        plt.title(f'order_parameter')

        # <editor-fold desc="create custom labels">
        indices = np.empty(self.numbers_heads, dtype=object)

        # Create a list of range objects based on the H values
        ranges = [range(H) for H in self.numbers_heads]

        # Use itertools.product to generate all combinations
        for combination in product(*ranges):
            # combination is a tuple containing values for h1, h2, ..., hL
            # You can access individual values like this:
            # h1, h2, h3, ..., hL = combination
            string = ""
            for i, index in enumerate(combination):
                if i == 0:  # do not put a "-" if this is the first index
                    string = string + f"{index}"
                else:
                    string = string + "-" + f"{index}"
            indices[combination] = string

        for l in range(len(self.numbers_heads) - 1):
            pre_arrangement = ""
            post_arrangement = ""
            for H in range(len(self.numbers_heads) - l):
                if H == 0:
                    pre_arrangement = f"h{H}"
                    post_arrangement = f"h{H})"
                elif H == 1:
                    pre_arrangement = f"h{H} " + pre_arrangement
                    post_arrangement = f"(h{H} " + post_arrangement
                else:
                    pre_arrangement = f"h{H} " + pre_arrangement
                    post_arrangement = f"h{H} " + post_arrangement
            indices = rearrange(indices, pre_arrangement + " -> " + post_arrangement)
        # </editor-fold>

        # Set custom ticks using the strings from the array
        plt.xticks(np.arange(len(indices)), indices, rotation=45, ha="right")
        plt.yticks(np.arange(len(indices)), indices)
        if plot_order_parameter_file_name is not None:
            plt.savefig(f'{plot_order_parameter_file_name}')
        else:
            plt.savefig('./order_parameters.png')

    def compute_renormalized_attentioned_input(self, symmetrized_order_parameter_largest):
        # self.attentioned_input is of size: [number_examples, input_width, total_head_size=H1*H2*...*HL]
        # and normalization: sqrt(variance_0)/sqrt(input_width * total_head_size)
        # effective_order_parameter_last_layer is of size [total_head_size,total_head_size]

        renormalized_attentioned_input = einsum(self.attentioned_input, symmetrized_order_parameter_largest,
                                                "b i H1, H1 H2 -> b i H2")

        return renormalized_attentioned_input

    def compute_mean_squared_readout_nonnormalized(self, renormalized_kernel, labels):
        # NOTE: this function does not add temperature. Temperature should be added manually to it's argument,
        # renormalized_kernel
        # NOTE: this function does not normalize the mean squared readout by 1/P

        # labels is of size [number_examples]
        # renormalized_kernel is of size [number_examples, number_examples]
        # we are computing y^T Gamma^-1 y (y:labels, gamma: renormalized kernel)

        # NOTE: this function synchronizes with CPU
        # NOTE: this is faster and more stable than computing the inverse
        inverse_kernel_dot_labels = torch.linalg.solve(renormalized_kernel, labels)

        mean_squared_readout = torch.dot(labels, inverse_kernel_dot_labels)

        return mean_squared_readout

    def compute_scalar_order_parameter(self, symmetrized_order_parameter_smallest):
        # CODING NOTE: # this function updates self.current_scalar_order_parameter with .detach().clone()

        # symmetrized_order_parameter_smallest is of size [hL, hL] (i.e. [# heads last layer, # heads last layer]
        # In other words, it is U_L

        # retrieve parameters
        var_readout = self.variances[-1]  # variance of the readout: sigma_a
        var_last_layer = self.variances[-2]  # variance of the last attention layer: sigma_L
        width_readout = self.model_widths[-1]  # width of the readout layer: N_a
        width_last_layer = self.model_widths[-2]  # width of the last attention layer: N_L
        n_heads = self.numbers_heads[-1]  # number of heads of the last attention layer: H_L

        # compute coefficients
        beta = torch.tensor(var_readout * (n_heads*width_last_layer/width_readout - 1))
        gamma = (torch.trace(symmetrized_order_parameter_smallest)
                 * (width_last_layer*var_readout)/(width_readout*var_last_layer))

        # compute scalar order parameter (i.e. U_a)
        scalar_order_parameter = (-beta + torch.sqrt(beta**2 + 4*gamma)) / 2

        # update current scalar order parameter
        self.current_scalar_order_parameter = scalar_order_parameter.detach().clone()

        return scalar_order_parameter

    def perform_hessian_test(self, training_labels):
        # WARNING: this method overwrites the stored training labels, and resets them to None at the end

        # store training labels, so they can be used below by self.compute_loss_action_for_hessian
        self.store_training_labels(training_labels.to(self.attentioned_input.device))

        # these are all order parameters stacked into a single vector. This is taken as an argument by
        # compute_loss_action_for_hessian
        packed_order_parameters = self.compute_packed_order_parameters().detach().clone()
        hessian = torch.autograd.functional.hessian(self.compute_loss_action_for_hessian, packed_order_parameters)
        hessian_eigenvalues = torch.linalg.eigvalsh(hessian)
        min_eig = torch.min(hessian_eigenvalues)

        print("\n")
        print("HESSIAN TEST: START")
        print("\n")
        print("Hessian eigenvalues: ")
        print(hessian_eigenvalues)
        print("\n")
        if min_eig > 0:
            print("Minimum found. Passed the Hessian test (all eigenvalues > 0)")
        if min_eig <= 0:
            print("WARNING: non minimum found. Failed the Hessian test (some eigenvalues <= 0)")
        print("\n")
        print("ESSIAN TEST: END")
        print("\n")

        # forget the stored training labels
        self.forget_training_labels()

    def return_renormalized_kernel(self):
        # we detach for safety because we don't want evaluation to be mixed with training for any reason
        order_parameter = self.compute_symmetrized_order_parameter_largest().detach().clone()

        kernel = self.compute_renormalized_kernel(order_parameter)

        return kernel

    @torch.no_grad()
    def compute_predictor_statistics(self, test_input, training_labels, gp_limit=False, order_param=None,
                                     force_unit_variance_gp=False, forced_temperature=None):
        # check that the size of the test input is consistent
        if self.input_width != test_input.size()[1]:
            print("ERROR: the width of the test input provided does not match with the input width of the model")
            sys.exit()

        if order_param is not None:
            order_parameter = order_param
        elif gp_limit:
            if force_unit_variance_gp:
                order_parameter = torch.eye(self.total_head_sizes[-1])
            else:
                tot_variance = 1
                for l in range(self.number_attention_layers + 1):
                    tot_variance *= self.variances[l + 1]  # the +1 is because we start from sigma_1, excluding sigma_0
                order_parameter = tot_variance * torch.eye(self.total_head_sizes[-1])
        else:
            # we detach for safety because we don't want evaluation to be mixed with training for any reason
            order_parameter = self.compute_symmetrized_order_parameter_largest().detach().clone()

        # this is the train-train kernel (e stand for "example")
        kernel_ee = self.compute_renormalized_kernel(order_parameter)
        # add temperature
        if forced_temperature is None:
            kernel_ee += torch.tensor(self.temperature)*torch.eye(kernel_ee.size()[0])
        else:
            print("I'm here using temperature:")
            print(forced_temperature)
            kernel_ee += torch.tensor(forced_temperature) * torch.eye(kernel_ee.size()[0])

        # now compute the test attentioned input
        attentioned_input_test = self.compute_attentioned_input(test_input)

        # compute the test-test e test-train kernels
        renormalized_attentioned_input_test = einsum(attentioned_input_test, order_parameter,
                                                     "test i H1, H1 H2 -> test i H2")

        kernel_et = einsum(self.attentioned_input, renormalized_attentioned_input_test,
                           "example i H, test i H -> example test")

        kernel_tt = einsum(renormalized_attentioned_input_test, attentioned_input_test,
                           "test i H, test i H -> test")
        # NOTE: Here the test-test kernel is just a vector of its diagonal! (we are not interested in
        # cross-correlations, just the variance of each test input)

        # COMPUTE PREDICTOR STATISTICS
        # compute (kernel_ee^-1) . kernel_et
        kernel_ee_inv_dot_kernel_et = torch.linalg.solve(kernel_ee, kernel_et)
        # size [# examples, # tests]

        # compute predictor mean
        predictor_mean = einsum(training_labels, kernel_ee_inv_dot_kernel_et, "example, example test -> test")

        # compute predictor variance
        # NOTE: this is just a vector, we are not interested in cross-correlations
        predictor_variance = kernel_tt - einsum(kernel_et, kernel_ee_inv_dot_kernel_et,
                                                "example test, example test -> test")

        return predictor_mean, predictor_variance

    # TODO:
    # IN THE FUTURE (optional): qk_internal dimension right now must be the same at every layer.
    # One could allow the user to specify its va


# TODO: implement correct naming of file
# TODO: summarize the conclusions so far










