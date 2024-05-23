import torch
import os
import numpy as np
import LIB_convergent_summation_heads as lib
import argparse
import time
import matplotlib.pyplot as plt


# <editor-fold desc="SELECT DEVICE">
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# </editor-fold>

# <editor-fold desc="PARSE ARGUMENTS">
# HYPERPARAMETERS
# DATA PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument("--number_examples", "-P", default=100, type=int)
parser.add_argument("--partial_width", "-N0", default=100, type=int)
parser.add_argument("--number_tokens", "-T", default=100, type=int)
parser.add_argument("--p_a_plus", default=0.9, type=float)
parser.add_argument("--p_a_minus", default=0.9, type=float)
parser.add_argument("--p_b_plus", default=0.1, type=float)
parser.add_argument("--p_b_minus", default=0.1, type=float)
parser.add_argument("--perpendicular_noise_strength", default=0.0, type=float,
                    help="noise perpendicular to the separating line. Noise in this direction disturbs the attention,"
                         "making the +/- states harder to discriminate")
parser.add_argument("--parallel_noise_strength", default=0.0, type=float,
                    help="noise parallel to the separating line. Noise in this direction only disturbs the value "
                         "weights, it is a special direction for which the disturbance is stronger")
parser.add_argument("--out_of_plane_noise_strength", default=0.0, type=float,
                    help="noise perpendicular to the plane in which v_plus and v_minus live. "
                         "Noise in this direction only disturbs the value weights, but still less than the parallel "
                         "direction")

# MODEL PARAMETERS
# model size hyperparameters
# parser.add_argument("--projected_input_dimension", "-N0", default=100, type=int) # maybe to implement in the future
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--model_widths", "-N", nargs='+', type=int, default=[1000],
                    help="single or multiple ints N1 Na")
parser.add_argument("--variances", nargs='+', type=float, default=[1000],
                    help="single or multiple floats sigma0 sigma1 sigma_a")
parser.add_argument("--seed", default=1, type=int)

# FIRST LAYER
# features_perturbation=0.0, positions_perturbation=0.0,
#                                               features_positions_cross_perturbation=0.0
parser.add_argument("--main_head_first_layer_style", type=str, default="same_token",
                    help="same_token, different_token, random_features, random_positions, all_random, "
                         "random_noninformative_only")
parser.add_argument("--main_head_first_layer_shift", type=float, default=-1.0,
                    help="positional shift for the main head")
parser.add_argument("--main_head_first_layer_beta", type=float, default=1.0,
                    help="inverse softmax-temperature")
parser.add_argument("--main_head_first_layer_seed", type=int)
parser.add_argument("--main_head_first_layer_features_perturbation", type=float, default=0.0,
                    help="perturbation on the informative features block of the attention weights")
parser.add_argument("--main_head_first_layer_positions_perturbation", type=float, default=0.0,
                    help="perturbation on the position block of the attention weights")
parser.add_argument("--main_head_first_layer_features_positions_cross_perturbation", type=float,
                    default=0.0)

parser.add_argument("--betas_uniform_attention_first_layer", nargs='+', type=float, default=[],
                    help="inverse softmax-temperatures for additional uniform attention heads "
                         "(sequence length indicates number of these heads")
parser.add_argument("--seeds_uniform_attention_first_layer", nargs='+', type=int, default=[])
parser.add_argument("--features_perturbation_uniform_attention_first_layer", nargs='+', type=float,
                    default=[],
                    help="informative features perturbation"
                         "(sequence length indicates number of these heads")
parser.add_argument("--positions_perturbation_uniform_attention_first_layer", nargs='+', type=float,
                    default=[],
                    help="positions perturbation"
                         "(sequence length indicates number of these heads")
parser.add_argument("--features_positions_cross_perturbation_uniform_attention_first_layer",
                    nargs='+', type=float,
                    default=[])

parser.add_argument("--shifts_same_token_heads_first_layer", nargs='+', type=float, default=[],
                    help="positional shifts for additional same token heads "
                         "(sequence length indicates number of these heads")
parser.add_argument("--betas_same_token_heads_first_layer", nargs='+', type=float, default=[],
                    help="inverse softmax-temperatures for additional same token heads "
                         "(sequence length indicates number of these heads")
parser.add_argument("--seeds_same_token_heads_first_layer", nargs='+', type=int, default=[])
parser.add_argument("--features_perturbation_same_token_heads_first_layer", nargs='+', type=float,
                    default=[],
                    help="informative features perturbation"
                         "(sequence length indicates number of these heads")
parser.add_argument("--positions_perturbation_same_token_heads_first_layer", nargs='+', type=float,
                    default=[],
                    help="positions perturbation"
                         "(sequence length indicates number of these heads")
parser.add_argument("--features_positions_cross_perturbation_same_token_heads_first_layer",
                    nargs='+', type=float,
                    default=[])

parser.add_argument("--shifts_different_token_heads_first_layer", nargs='+', type=float, default=[],
                    help="positional shifts for additional different token heads "
                         "(sequence length indicates number of these heads")
parser.add_argument("--betas_different_token_heads_first_layer", nargs='+', type=float, default=[],
                    help="inverse softmax-temperatures for additional different token heads "
                         "(sequence length indicates number of these heads")
parser.add_argument("--seeds_different_token_heads_first_layer", nargs='+', type=int, default=[])
parser.add_argument("--features_perturbation_different_token_heads_first_layer", nargs='+', type=float,
                    default=[],
                    help="informative features perturbation"
                         "(sequence length indicates number of these heads")
parser.add_argument("--positions_perturbation_different_token_heads_first_layer", nargs='+', type=float,
                    default=[],
                    help="positions perturbation"
                         "(sequence length indicates number of these heads")
parser.add_argument("--features_positions_cross_perturbation_different_token_heads_first_layer",
                    nargs='+', type=float,
                    default=[])

parser.add_argument("--betas_blank_attention_first_layer", nargs='+', type=float, default=[],
                    help="inverse softmax-temperatures for additional uniform attention heads "
                         "(sequence length indicates number of these heads")
parser.add_argument("--seeds_blank_attention_first_layer", nargs='+', type=int, default=[])
parser.add_argument("--features_perturbation_blank_attention_first_layer", nargs='+', type=float,
                    default=[],
                    help="informative features perturbation"
                         "(sequence length indicates number of these heads")
parser.add_argument("--positions_perturbation_blank_attention_first_layer", nargs='+', type=float,
                    default=[],
                    help="positions perturbation"
                         "(sequence length indicates number of these heads")
parser.add_argument("--features_positions_cross_perturbation_blank_attention_first_layer",
                    nargs='+', type=float,
                    default=[])

# SECOND LAYER
parser.add_argument("--main_head_second_layer_style", type=str, default="same_token",
                    help="same_token, different_token, random_features, random_positions, all_random, "
                         "random_noninformative_only")
parser.add_argument("--main_head_second_layer_shift", type=float, default=-1.0,
                    help="positional shift for the main head")
parser.add_argument("--main_head_second_layer_beta", type=float, default=1.0,
                    help="inverse softmax-temperature")
parser.add_argument("--main_head_second_layer_seed", type=int)
parser.add_argument("--main_head_second_layer_features_perturbation", type=float, default=0.0,
                    help="perturbation on the informative features block of the attention weights")
parser.add_argument("--main_head_second_layer_positions_perturbation", type=float, default=0.0,
                    help="perturbation on the position block of the attention weights")
parser.add_argument("--main_head_second_layer_features_positions_cross_perturbation", type=float,
                    default=0.0)

parser.add_argument("--betas_uniform_attention_second_layer", nargs='+', type=float, default=[],
                    help="inverse softmax-temperatures for additional uniform attention heads "
                         "(sequence length indicates number of these heads")
parser.add_argument("--seeds_uniform_attention_second_layer", nargs='+', type=int, default=[])
parser.add_argument("--features_perturbation_uniform_attention_second_layer", nargs='+', type=float,
                    default=[],
                    help="informative features perturbation"
                         "(sequence length indicates number of these heads")
parser.add_argument("--positions_perturbation_uniform_attention_second_layer", nargs='+', type=float,
                    default=[],
                    help="positions perturbation"
                         "(sequence length indicates number of these heads")
parser.add_argument("--features_positions_cross_perturbation_uniform_attention_second_layer",
                    nargs='+', type=float,
                    default=[])

parser.add_argument("--shifts_same_token_heads_second_layer", nargs='+', type=float, default=[],
                    help="positional shifts for additional same token heads "
                         "(sequence length indicates number of these heads")
parser.add_argument("--betas_same_token_heads_second_layer", nargs='+', type=float, default=[],
                    help="inverse softmax-temperatures for additional same token heads "
                         "(sequence length indicates number of these heads")
parser.add_argument("--seeds_same_token_heads_second_layer", nargs='+', type=int, default=[])
parser.add_argument("--features_perturbation_same_token_heads_second_layer", nargs='+', type=float,
                    default=[],
                    help="informative features perturbation"
                         "(sequence length indicates number of these heads")
parser.add_argument("--positions_perturbation_same_token_heads_second_layer", nargs='+', type=float,
                    default=[],
                    help="positions perturbation"
                         "(sequence length indicates number of these heads")
parser.add_argument("--features_positions_cross_perturbation_same_token_heads_second_layer",
                    nargs='+', type=float,
                    default=[])

parser.add_argument("--shifts_different_token_heads_second_layer", nargs='+', type=float, default=[],
                    help="positional shifts for additional different token heads "
                         "(sequence length indicates number of these heads")
parser.add_argument("--betas_different_token_heads_second_layer", nargs='+', type=float, default=[],
                    help="inverse softmax-temperatures for additional different token heads "
                         "(sequence length indicates number of these heads")
parser.add_argument("--seeds_different_token_heads_second_layer", nargs='+', type=int, default=[])
parser.add_argument("--features_perturbation_different_token_heads_second_layer", nargs='+', type=float,
                    default=[],
                    help="informative features perturbation"
                         "(sequence length indicates number of these heads")
parser.add_argument("--positions_perturbation_different_token_heads_second_layer", nargs='+', type=float,
                    default=[],
                    help="positions perturbation"
                         "(sequence length indicates number of these heads")
parser.add_argument("--features_positions_cross_perturbation_different_token_heads_second_layer",
                    nargs='+', type=float,
                    default=[])

parser.add_argument("--betas_blank_attention_second_layer", nargs='+', type=float, default=[],
                    help="inverse softmax-temperatures for additional uniform attention heads "
                         "(sequence length indicates number of these heads")
parser.add_argument("--seeds_blank_attention_second_layer", nargs='+', type=int, default=[])
parser.add_argument("--features_perturbation_blank_attention_second_layer", nargs='+', type=float,
                    default=[],
                    help="informative features perturbation"
                         "(sequence length indicates number of these heads")
parser.add_argument("--positions_perturbation_blank_attention_second_layer", nargs='+', type=float,
                    default=[],
                    help="positions perturbation"
                         "(sequence length indicates number of these heads")
parser.add_argument("--features_positions_cross_perturbation_blank_attention_second_layer",
                    nargs='+', type=float,
                    default=[])

# TRAINING PARAMETERS
parser.add_argument("--learning_rate", "-lr", default=0.001, type=float)
parser.add_argument("--gradient_tolerance", "-tol", default=1e-06, type=float,
                    help="at what size of gradients to stop the training")
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--scheduler_patience", default=1000, type=int,
                    help="number of epochs the scheduler should wait before decreasing the learning rate")
parser.add_argument("--lr_reduce_factor", default=0.1, type=float,
                    help="factor of reduction of the learning rate")
parser.add_argument("--order_parameter_perturbation_strength", default=1.0, type=float)
parser.add_argument("--order_parameter_scale", default=1.0, type=float)
parser.add_argument("--order_parameter_seed", default=1, type=int)

# COMPUTATION PARAMETERS
parser.add_argument("--single_precision", action="store_true")
parser.add_argument("--force_cpu", action="store_true")

# OUTPUT PARAMETERS
parser.add_argument("--dont_store_scalars", action="store_true")

parser.add_argument("--number_steps_store_scalars", default=1, type=int,
                    help="defines every how many steps the scalar training statistics (e.g. loss, max abs gradient, "
                         "avg order parameter, etc...) are stored")

parser.add_argument("--dont_store_tensors", action="store_true")

parser.add_argument("--number_steps_store_tensors", default=100, type=int,
                    help="defines every how many steps the tensor training statistics (e.g. gradients vector, "
                         "order parameters, etc...) are stored")

parser.add_argument("--dont_store_checkpoint", action="store_true")

parser.add_argument("--number_steps_store_checkpoint", default=100, type=int,
                    help="defines every how many steps the tensor training statistics (e.g. gradients vector, "
                         "order parameters, etc...) are stored")

parser.add_argument("--number_steps_print_info", default=100, type=int,
                    help="defines every how many steps the training statistics (e.g. gradients vector, "
                         "order parameters, etc...) are printed")


# LOCATIONS
parser.add_argument('--results_storage_location', type=str, default="./",
                    help='where the simulation results and checkpoints will be stored')
parser.add_argument('--results_id', type=str, default="",
                    help='A user-defined identifier for the stored results, that is appended to the generic '
                         'storage file name. Useful for example to not mix checkpoints of different simulations')

# OTHER OPTIONS
parser.add_argument("--test_kernel_invertibility", action="store_true",
                    help="This skips the training and just tests the invertibility of the Kernel")
parser.add_argument("--plot_kernel_invertibility", action="store_true",
                    help="Plot the results of testing the invertibility of the Kernel")
parser.add_argument("--test_kernel_invertibility_with_temperature", action="store_true",
                    help="Tests the gp kernel invertibility with the addition of temperature*identity")

parser.add_argument("--plot_pre_kernels", action="store_true",
                    help="This skips the training and just plots the pre-kernels")
parser.add_argument("--plot_pre_kernels_style", type=str, default="all",
                    help="all, diagonal. Whether to plot only the diagonal or also the off-diagonal pre-kernels")
parser.add_argument("--plot_pre_kernels_with_temperature", action="store_true",
                    help="plots the pre_kernels with the addition of temperature*identity")

parser.add_argument("--plot_kernel", action="store_true")
parser.add_argument("--plot_kernel_with_temperature", action="store_true",
                    help="plots the kernel with the addition of temperature*identity")

parser.add_argument("--hessian_test", action="store_true",
                    help="perform the hessian test to check if we are at a minimum")


args = parser.parse_args()
# </editor-fold>

# <editor-fold desc="FORCE FLAGS">
# IMPLEMENT SOME OF THE FLAGS:
if args.single_precision:
    torch.set_default_dtype(torch.float32)
else:
    torch.set_default_dtype(torch.float64)

if args.force_cpu:
    device = "cpu"
# </editor-fold>

print(f"\nDEVICE IDENTIFICATION: START")
print(f"\nUsing {device} device")
print(f"\nDefault precision: {torch.get_default_dtype()}")
print(f"\nDEVICE IDENTIFICATION: END")
print("\n")

# <editor-fold desc="CONSTRUCT GENERAL STORAGE FILENAME">
# construct the general storage file name
string_list = [str(element) for element in args.variances]
delimiter = "_"
variances_list = delimiter.join(string_list)
string_list = [str(element) for element in args.model_widths]
delimiter = "_"
model_widths_list = delimiter.join(string_list)
general_script_name = "_conv_sum_heads_bin_reg_MARKOV_OPTION_D_theory"
general_storage_file_name = (f"_P{args.number_examples}_N0_{args.partial_width}_T_{args.number_tokens}_"
                             f"N{model_widths_list}_"
                             f"ParNoise{args.parallel_noise_strength}_PerpNoise{args.perpendicular_noise_strength}_"
                             f"OopNoise{args.out_of_plane_noise_strength}"
                             f"_temp{args.temperature}_"
                             f"var{variances_list}_"
                             f"ps{args.p_a_plus}_{args.p_a_minus}_{args.p_b_plus}_{args.p_b_minus}")
checkpoint_file_name = (args.results_storage_location + args.results_id + "_CHECKPOINT" + general_script_name
                        + general_storage_file_name + ".pkl")
# </editor-fold>

# START TIME
start_time = time.time()

# <editor-fold desc="INITIALIZE TRAINING DATA">
# initialize the training data
dataset_info = {
    "dataset": "markov_optionE",
    "number_examples": args.number_examples,
    "partial_width": args.partial_width,
    "number_tokens": args.number_tokens,
    "p_a_plus": args.p_a_plus,
    "p_a_minus": args.p_a_minus,
    "p_b_plus": args.p_b_plus,
    "p_b_minus": args.p_b_minus,
    "max_number_tokens": args.number_tokens,
    "perpendicular_noise_strength": args.perpendicular_noise_strength,
    "parallel_noise_strength": args.parallel_noise_strength,
    "out_of_plane_noise_strength": args.out_of_plane_noise_strength,
    "v_minus": None,
    "v_plus": None,
    }

input, labels, dataset_info = lib.prepare_dataset(None, dataset_info, train=True)
# </editor-fold>

# <editor-fold desc="CONSTRUCT THE ATTENTION WEIGHTS">

# <editor-fold desc="FIRST LAYER">
# initialize the main head
lib.seed_everything(seed=args.main_head_first_layer_seed)
w_weights_layer1 = (lib.generate_attention_weights_markov_optionE(dataset_info_after_initialization=dataset_info,
                                                                  shift=args.main_head_first_layer_shift,
                                                                  version=args.main_head_first_layer_style,
                                                                  features_perturbation=
                                                                  args.main_head_first_layer_features_perturbation,
                                                                  positions_perturbation=
                                                                  args.main_head_first_layer_positions_perturbation,
                                                                  features_positions_cross_perturbation=
                                                                  args.main_head_first_layer_features_positions_cross_perturbation)
                    * args.main_head_first_layer_beta)
# create list of heads style info
heads_style_info_layer1 = [args.main_head_first_layer_style
                           + "_shift_" + f"{args.main_head_first_layer_shift}"
                           + "_beta_" + f"{args.main_head_first_layer_beta}"
                           + "_Ipert_" + f"{args.main_head_first_layer_features_perturbation}"
                           + "_Ppert_" + f"{args.main_head_first_layer_positions_perturbation}"
                           + "_Opert_" + f"{args.main_head_second_layer_features_positions_cross_perturbation}"]
for (seed, shift, beta, Ipert, Ppert, Opert) in zip(args.seeds_same_token_heads_first_layer,
                                       args.shifts_same_token_heads_first_layer,
                                       args.betas_same_token_heads_first_layer,
                                       args.features_perturbation_same_token_heads_first_layer,
                                       args.positions_perturbation_same_token_heads_first_layer,
                                       args.features_positions_cross_perturbation_same_token_heads_first_layer):
    lib.seed_everything(seed=seed)
    new_weights = lib.generate_attention_weights_markov_optionE(dataset_info_after_initialization=dataset_info,
                                                                shift=shift,
                                                                version="same_token",
                                                                features_perturbation=Ipert,
                                                                positions_perturbation=Ppert,
                                                                features_positions_cross_perturbation=Opert) * beta
    w_weights_layer1 = torch.cat([w_weights_layer1, new_weights], dim=0)
    heads_style_info_layer1.append("same_token"
                                   + "_shift_" + f"{shift}"
                                   + "_beta_" + f"{beta}"
                                   + "_Ipert_" + f"{Ipert}"
                                   + "_Ppert_" + f"{Ppert}"
                                   + "_Opert_" + f"{Opert}")
for (seed, shift, beta, Ipert, Ppert, Opert) in zip(args.seeds_different_token_heads_first_layer,
                                       args.shifts_different_token_heads_first_layer,
                                       args.betas_different_token_heads_first_layer,
                                       args.features_perturbation_different_token_heads_first_layer,
                                       args.positions_perturbation_different_token_heads_first_layer,
                                       args.features_positions_cross_perturbation_different_token_heads_first_layer
                                                            ):
    lib.seed_everything(seed=seed)
    new_weights = lib.generate_attention_weights_markov_optionE(dataset_info_after_initialization=dataset_info,
                                                                shift=shift,
                                                                version="different_token",
                                                                features_perturbation=Ipert,
                                                                positions_perturbation=Ppert,
                                                                features_positions_cross_perturbation=Opert) * beta
    w_weights_layer1 = torch.cat([w_weights_layer1, new_weights], dim=0)
    heads_style_info_layer1.append("different_token"
                                   + "_shift_" + f"{shift}"
                                   + "_beta_" + f"{beta}"
                                   + "_Ipert_" + f"{Ipert}"
                                   + "_Ppert_" + f"{Ppert}"
                                   + "_Opert_" + f"{Opert}")
for (seed, beta, Ipert, Ppert, Opert) in zip(args.seeds_uniform_attention_first_layer,
                                       args.betas_uniform_attention_first_layer,
                                       args.features_perturbation_uniform_attention_first_layer,
                                       args.positions_perturbation_uniform_attention_first_layer,
                                       args.features_positions_cross_perturbation_uniform_attention_first_layer):
    lib.seed_everything(seed=seed)
    new_weights = lib.generate_attention_weights_markov_optionE(dataset_info_after_initialization=dataset_info,
                                                                version="uniform_attention",
                                                                features_perturbation=Ipert,
                                                                positions_perturbation=Ppert,
                                                                features_positions_cross_perturbation=Opert) * beta
    w_weights_layer1 = torch.cat([w_weights_layer1, new_weights], dim=0)
    heads_style_info_layer1.append("uniform_attention"
                                   + "_beta_" + f"{beta}"
                                   + "_Ipert_" + f"{Ipert}"
                                   + "_Ppert_" + f"{Ppert}"
                                   + "_Opert_" + f"{Opert}")
for (seed, beta, Ipert, Ppert, Opert) in zip(args.seeds_blank_attention_first_layer,
                                       args.betas_blank_attention_first_layer,
                                       args.features_perturbation_blank_attention_first_layer,
                                       args.positions_perturbation_blank_attention_first_layer,
                                       args.features_positions_cross_perturbation_blank_attention_first_layer):
    lib.seed_everything(seed=seed)
    new_weights = lib.generate_attention_weights_markov_optionE(dataset_info_after_initialization=dataset_info,
                                                                version="blank",
                                                                features_perturbation=Ipert,
                                                                positions_perturbation=Ppert,
                                                                features_positions_cross_perturbation=Opert) * beta
    w_weights_layer1 = torch.cat([w_weights_layer1, new_weights], dim=0)
    heads_style_info_layer1.append("blank"
                                   + "_beta_" + f"{beta}"
                                   + "_Ipert_" + f"{Ipert}"
                                   + "_Ppert_" + f"{Ppert}"
                                   + "_Opert_" + f"{Opert}")

# </editor-fold>

# <editor-fold desc="SECOND LAYER">
# initialize the main head
lib.seed_everything(seed=args.main_head_second_layer_seed)
w_weights_layer2 = (lib.generate_attention_weights_markov_optionE(dataset_info_after_initialization=dataset_info,
                                                                  shift=args.main_head_second_layer_shift,
                                                                  version=args.main_head_second_layer_style,
                                                                  features_perturbation=
                                                                  args.main_head_second_layer_features_perturbation,
                                                                  positions_perturbation=
                                                                  args.main_head_second_layer_positions_perturbation,
                                                                  features_positions_cross_perturbation=
                                                                  args.main_head_second_layer_features_positions_cross_perturbation)
                    * args.main_head_second_layer_beta)
# create list of heads style info
heads_style_info_layer2 = [args.main_head_second_layer_style
                           + "_shift_" + f"{args.main_head_second_layer_shift}"
                           + "_beta_" + f"{args.main_head_second_layer_beta}"
                           + "_Ipert_" + f"{args.main_head_second_layer_features_perturbation}"
                           + "_Ppert_" + f"{args.main_head_second_layer_positions_perturbation}"
                           + "_Opert_" + f"{args.main_head_second_layer_features_positions_cross_perturbation}"]
for (seed, shift, beta, Ipert, Ppert, Opert) in zip(args.seeds_same_token_heads_second_layer,
                                       args.shifts_same_token_heads_second_layer,
                                       args.betas_same_token_heads_second_layer,
                                       args.features_perturbation_same_token_heads_second_layer,
                                       args.positions_perturbation_same_token_heads_second_layer,
                                       args.features_positions_cross_perturbation_same_token_heads_second_layer):
    lib.seed_everything(seed=seed)
    new_weights = lib.generate_attention_weights_markov_optionE(dataset_info_after_initialization=dataset_info,
                                                                shift=shift,
                                                                version="same_token",
                                                                features_perturbation=Ipert,
                                                                positions_perturbation=Ppert,
                                                                features_positions_cross_perturbation=Opert) * beta
    w_weights_layer2 = torch.cat([w_weights_layer2, new_weights], dim=0)
    heads_style_info_layer2.append("same_token"
                                   + "_shift_" + f"{shift}"
                                   + "_beta_" + f"{beta}"
                                   + "_Ipert_" + f"{Ipert}"
                                   + "_Ppert_" + f"{Ppert}"
                                   + "_Opert_" + f"{Ppert}")
for (seed, shift, beta, Ipert, Ppert, Opert) in zip(args.seeds_different_token_heads_second_layer,
                                                            args.shifts_different_token_heads_second_layer,
                                       args.betas_different_token_heads_second_layer,
                                       args.features_perturbation_different_token_heads_second_layer,
                                       args.positions_perturbation_different_token_heads_second_layer,
                                       args.features_positions_cross_perturbation_different_token_heads_second_layer):
    lib.seed_everything(seed=seed)
    new_weights = lib.generate_attention_weights_markov_optionE(dataset_info_after_initialization=dataset_info,
                                                                shift=shift,
                                                                version="different_token",
                                                                features_perturbation=Ipert,
                                                                positions_perturbation=Ppert,
                                                                features_positions_cross_perturbation=Opert) * beta
    w_weights_layer2 = torch.cat([w_weights_layer2, new_weights], dim=0)
    heads_style_info_layer2.append("different_token"
                                   + "_shift_" + f"{shift}"
                                   + "_beta_" + f"{beta}"
                                   + "_Ipert_" + f"{Ipert}"
                                   + "_Ppert_" + f"{Ppert}"
                                   + "_Opert_" + f"{Ppert}")
for (seed, beta, Ipert, Ppert, Opert) in zip(args.seeds_uniform_attention_second_layer,
                                             args.betas_uniform_attention_second_layer,
                                             args.features_perturbation_uniform_attention_second_layer,
                                             args.positions_perturbation_uniform_attention_second_layer,
                                             args.features_positions_cross_perturbation_uniform_attention_second_layer):
    lib.seed_everything(seed=seed)
    new_weights = lib.generate_attention_weights_markov_optionE(dataset_info_after_initialization=dataset_info,
                                                                version="uniform_attention",
                                                                features_perturbation=Ipert,
                                                                positions_perturbation=Ppert,
                                                                features_positions_cross_perturbation=Opert) * beta
    w_weights_layer2 = torch.cat([w_weights_layer2, new_weights], dim=0)
    heads_style_info_layer2.append("uniform_attention"
                                   + "_beta_" + f"{beta}"
                                   + "_Ipert_" + f"{Ipert}"
                                   + "_Ppert_" + f"{Ppert}"
                                   + "_Opert_" + f"{Ppert}")
for (seed, beta, Ipert, Ppert, Opert) in zip(args.seeds_blank_attention_second_layer,
                                       args.betas_blank_attention_second_layer,
                                       args.features_perturbation_blank_attention_second_layer,
                                       args.positions_perturbation_blank_attention_second_layer,
                                       args.features_positions_cross_perturbation_blank_attention_second_layer):
    lib.seed_everything(seed=seed)
    new_weights = lib.generate_attention_weights_markov_optionE(dataset_info_after_initialization=dataset_info,
                                                                version="blank",
                                                                features_perturbation=Ipert,
                                                                positions_perturbation=Ppert,
                                                                features_positions_cross_perturbation=Opert) * beta
    w_weights_layer2 = torch.cat([w_weights_layer2, new_weights], dim=0)
    heads_style_info_layer2.append("blank"
                                   + "_beta_" + f"{beta}"
                                   + "_Ipert_" + f"{Ipert}"
                                   + "_Ppert_" + f"{Ppert}"
                                   + "_Opert_" + f"{Ppert}")

# </editor-fold>

# DEBUG: START
print("w_weights_layer2.size()")
print(w_weights_layer2.size())
print("w_weights_layer1.size()")
print(w_weights_layer1.size())

heads_style_info = [heads_style_info_layer1, heads_style_info_layer2]

# unseed everything
lib.unseed_everything()
# </editor-fold>

# <editor-fold desc="INITIALIZE MODEL">
# initialize the model...
# ...from scratch
if not os.path.isfile(checkpoint_file_name):
    # if no checkpoint is present, the starting epoch should be 0
    starting_epoch = 0

    # retrieve some info to initialize the model
    input_width = input.size()[1]
    numbers_heads = [w_weights_layer1.size()[0], w_weights_layer2.size()[0]]  # this should be a list
    number_attention_layers = 2
    model = lib.ConvergentSummationHeads(numbers_heads, args.model_widths, number_attention_layers,
                                         input_width, args.variances, attention_nonlinearity="softmax",
                                         temperature=args.temperature, token_readout_style="last_token")
    # set attention weights
    model.set_w([w_weights_layer1, w_weights_layer2])
    model.store_heads_style_info(heads_style_info)

    # initialize order parameters
    model.set_order_parameters_gp_perturbed(perturbation_strength=args.order_parameter_perturbation_strength,
                                            seed=args.order_parameter_seed, scale=args.order_parameter_scale)

# ...from checkpoint
else:
    checkpoint = torch.load(checkpoint_file_name)
    # retrieve the model
    model = checkpoint["model"]
    # retrieve the current training_results
    train_results = checkpoint["train_results"]
    # set the starting epoch from where the checkpoint was saved
    starting_epoch = checkpoint["reached_epoch"] + 1
    print(f"\nFOUND A CHECKPOINT.\nStarting from epoch {starting_epoch}")
    print("\n")
    print("MODEL INFO: START")
    print("\n")
    model.print_architecture()
    print("\n")
    print("MODEL INFO: END")
    print("\n")

# </editor-fold>

# <editor-fold desc="SETUP OPTIMIZER, MODEL, DATASET FOR TRAINING">
# load the model for training
model.load(input, dataset_info)

# send everything to device
model.to_device(device)  # this is a special function created for the model to map all that is needed to a device
labels = labels.to(device)

# initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# initialize the learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_reduce_factor,
                                                       patience=args.scheduler_patience, min_lr=0.001)

# </editor-fold>

# <editor-fold desc="test invertibility of GP Kernel (optional, exits the script afterward)">
if args.test_kernel_invertibility:
    model.test_gp_kernel_invertibility(plot=args.plot_kernel_invertibility,
                                       with_temperature=args.test_kernel_invertibility_with_temperature)
    exit()
# </editor-fold>

# <editor-fold desc="plot pre-kernels (optional, exits the script afterward)">
if args.plot_pre_kernels:
    model.plot_pre_kernels(style=args.plot_pre_kernels_style, with_temperature=args.plot_pre_kernels_with_temperature)
if args.plot_kernel:
    model.plot_kernel(with_temperature=args.plot_kernel_with_temperature)
if args.plot_pre_kernels or args.plot_kernel:
    plt.show()
    exit()
# </editor-fold>

# <editor-fold desc="INITIALIZE TRAINING INFO CONTAINERS">

# this should only be done if no checkpoint is retrieved. If a checkpoint is retrieved, instead, we will have also
# retrieved an already initialized and partially filled train_results dictionary
if not os.path.isfile(checkpoint_file_name):
    # INITIALIZE LISTS OF TRAINING INFO
    train_results = {}
    train_results["loss_history"] = []
    train_results["loss_energy_history"] = []
    train_results["loss_entropy_history"] = []
    train_results["time_points_history"] = []
    # the lists below are initialized as lists of lists,
    # e.g. containing [list_max_grad UL, list_max_grad UL-1, etc...]
    train_results["max_abs_gradient_history"] = []
    train_results["avg_abs_gradient_history"] = []
    train_results["energy_max_abs_gradient_history"] = []
    train_results["energy_avg_abs_gradient_history"] = []
    train_results["entropy_max_abs_gradient_history"] = []
    train_results["entropy_avg_abs_gradient_history"] = []
    for i in range(model.number_attention_layers):
        train_results["max_abs_gradient_history"].append([])
        train_results["avg_abs_gradient_history"].append([])
        train_results["energy_max_abs_gradient_history"].append([])
        train_results["energy_avg_abs_gradient_history"].append([])
        train_results["entropy_max_abs_gradient_history"].append([])
        train_results["entropy_avg_abs_gradient_history"].append([])
    train_results["order_parameters_history"] = []
    for i in range(model.number_attention_layers + 1):
        # the +1 is because we also want Ua
        train_results["order_parameters_history"].append([])
# </editor-fold>

# <editor-fold desc="TRAIN LOOP">
achieved_tolerance = False
last_lr = args.learning_rate
for t in range(starting_epoch, args.epochs):
    # ensure the gradients are set to zero before computing the next gradient
    optimizer.zero_grad()

    # compute cost
    energy, entropy = model.compute_loss_action(labels, return_energy_entropy=True)

    # Compute gradients separately for entropy term
    energy.backward(retain_graph=True)  # Retain graph to compute entropy gradients further below

    # <editor-fold desc="Store gradients for energy term">
    energy_gradients = []
    for p in model.parameters():
        if p.grad is None:
            # If a parameter does not appear in the cost function, pytorch stores its gradient as None,
            # instead of a tensor of zeros to save memory.
            # Overwrite this with a tensor of zeros instead.
            energy_gradients.append(torch.zeros(p.size(), device=p.device))
        else:
            energy_gradients.append(p.grad.clone())
    # </editor-fold>

    # Zero gradients before computing entropy gradients
    optimizer.zero_grad()

    # Compute gradients separately for entropy term
    entropy.backward()

    # <editor-fold desc="Store gradients for entropy term">
    entropy_gradients = []
    for p in model.parameters():
        if p.grad is None:
            # If a parameter does not appear in the cost function, pytorch stores its gradient as None,
            # instead of a tensor of zeros to save memory.
            # Overwrite this with a tensor of zeros instead.
            entropy_gradients.append(torch.zeros(p.size(), device=p.device))
        else:
            entropy_gradients.append(p.grad.clone())
    # </editor-fold>

    # Update gradients as the sum of the two gradients
    for param, energy_grad, entropy_grad in zip(model.parameters(), energy_gradients, entropy_gradients):
        param.grad = energy_grad + entropy_grad

    # <editor-fold desc="check whether we achieved tolerance">
    max_gradients = []
    for l, param in enumerate(model.parameters()):
        max_abs_grad = torch.max(torch.abs(param.grad)).detach().cpu().clone().numpy()
        max_gradients.append(max_abs_grad)
    max_grad = np.max(max_gradients)
    if max_grad < args.gradient_tolerance:
        print(f"Achieved required tolerance, terminating training at step {t}")
        achieved_tolerance = True
        # break
    # </editor-fold>

    # <editor-fold desc="store scalar training info">
    if (((t + 1) % args.number_steps_store_scalars == 0 or t == 0 or t == (args.epochs - 1) or achieved_tolerance)
            and not args.dont_store_scalars):
        train_results["time_points_history"].append(t)
        loss = energy + entropy
        train_results["loss_history"].append(loss.item())
        train_results["loss_energy_history"].append(energy.item())
        train_results["loss_entropy_history"].append(entropy.item())

        for l, (param, energy_grad, entropy_grad) in (
                enumerate(zip(model.parameters(), energy_gradients, entropy_gradients))):
            # store total grad info
            max_abs_grad = torch.max(torch.abs(param.grad)).detach().cpu().clone().numpy()
            train_results["max_abs_gradient_history"][l].append(max_abs_grad)
            avg_abs_grad = torch.mean(torch.abs(param.grad)).detach().cpu().clone().numpy()
            train_results["avg_abs_gradient_history"][l].append(avg_abs_grad)
            # store energy grad info
            energy_max_abs_grad = torch.max(torch.abs(energy_grad)).detach().cpu().clone().numpy()
            train_results["energy_max_abs_gradient_history"][l].append(energy_max_abs_grad)
            energy_avg_abs_grad = torch.mean(torch.abs(energy_grad)).detach().cpu().clone().numpy()
            train_results["energy_avg_abs_gradient_history"][l].append(energy_avg_abs_grad)
            # store entropy info
            entropy_max_abs_grad = torch.max(torch.abs(entropy_grad)).detach().cpu().clone().numpy()
            train_results["entropy_max_abs_gradient_history"][l].append(entropy_max_abs_grad)
            entropy_avg_abs_grad = torch.mean(torch.abs(entropy_grad)).detach().cpu().clone().numpy()
            train_results["entropy_avg_abs_gradient_history"][l].append(entropy_avg_abs_grad)

    # </editor-fold>

    # <editor-fold desc="store tensorial training info">
    if (((t + 1) % args.number_steps_store_tensors == 0 or t == 0 or t == (args.epochs - 1) or achieved_tolerance)
            and not args.dont_store_tensors):
        train_results["order_parameters_history"][0].append(model.current_scalar_order_parameter.cpu().clone().numpy())
        for l, param in enumerate(model.parameters()):
            train_results["order_parameters_history"][l + 1].append(param.detach().cpu().clone().numpy())
    # </editor-fold>

    # <editor-fold desc="store checkpoint">
    if (t + 1) % args.number_steps_store_checkpoint == 0 and not args.dont_store_checkpoint:
        # unload the model, so we do not store useless data
        attentioned_input = model.unload_before_checkpoint()
        # save
        torch.save({
            "reached_epoch": t,
            "model": model,
            "dataset_info": dataset_info,
            "train_results": train_results,
        }
            , checkpoint_file_name)
        # reload the model to continue training
        model.load_after_checkpoint(attentioned_input)
    # </editor-fold>

    # PRINT TRAINING INFO
    if (t + 1) % args.number_steps_print_info == 0 or t == 0 or t == (args.epochs - 1):
        loss = energy + entropy
        print(f"\nLoss at step {t}/{args.epochs}")
        print(f"Total: {loss.item():>7f} (Energy: {energy.item():>7f}; Entropy: {entropy.item():>7f})")
        print(f"max abs gradient: {max_grad}")

    # exit training if we achieved tolerance
    if achieved_tolerance:
        break

    # do one optimization step (unless we are at the last step)
    if t != (args.epochs - 1):
        optimizer.step()

    # do one step of the learning_rate scheduler
    scheduler.step(max_grad)
    current_lr = scheduler.get_last_lr()[0]
    if current_lr != last_lr:
        print(f"Updated lr from {last_lr} to {current_lr}")
    last_lr = current_lr

# </editor-fold>

# <editor-fold desc="PRINT END-OF-TRAINING INFO">
print("\nEND-OF-TRAINING-INFO: START\n")
# print loss info
print(f"Final loss at step {train_results['time_points_history'][-1]}/{args.epochs}:")
print(f"Total: {train_results['loss_history'][-1]:>7f} (Energy: {train_results['loss_energy_history'][-1]:>7f}; "
      f"Entropy: {train_results['loss_entropy_history'][-1]:>7f})")
# print max gradient info
max_gradients = []
for l, param in enumerate(model.parameters()):
    max_abs_grad = torch.max(torch.abs(param.grad)).detach().cpu().clone().numpy()
    max_gradients.append(max_abs_grad)
max_grad = np.max(max_gradients)
print(f"max abs gradient: {max_grad}")
# print final order parameter info
final_order_param = model.order_parameters[-1].detach().cpu().clone()
size = final_order_param.size()[0]
off_diagonals = final_order_param[torch.triu(torch.ones(size, size)) == 1]
diagonals = torch.diag(final_order_param)
mean_off_diagonals = torch.mean(torch.abs(off_diagonals))
std_off_diagonals = torch.std(torch.abs(off_diagonals))
mean_diagonals = torch.mean(torch.abs(diagonals))
std_diagonals = torch.std(torch.abs(diagonals))
print(f"ORDER PARAMETER INFO:")
print(f"mean(abs) diagonal: {mean_diagonals}")
print(f"std(abs) diagonal: {std_diagonals}")
print(f"mean(abs) off-diagonal: {mean_off_diagonals}")
print(f"std(abs) off-diagonal: {std_off_diagonals}")
print("\nEND-OF-TRAINING-INFO: END\n")
# </editor-fold>

# <editor-fold desc="CHECK THE HESSIAN (optional)">
if args.hessian_test:
    model.perform_hessian_test(labels)
# </editor-fold>

# unload the model after training
model.unload()

# <editor-fold desc="PRINT TIME AND MEMORY USAGE INFO">
print(f"\nSCRIPT PERFORMANCE INFO: START")
max_gpu_memory_allocated = torch.cuda.max_memory_allocated(device="cuda")
print("\nMaximum memory allocated:")
print(f"GPU: {max_gpu_memory_allocated/10**9} GB ({max_gpu_memory_allocated/10**6} MB)")
time_elapsed = time.time() - start_time
print("\ntotal running time (mins):")
print(time_elapsed / 60)
print("\ntotal running time (hrs):")
print(time_elapsed / 3600)
print(f"\nSCRIPT PERFORMANCE INFO: END")
print("\n")
# </editor-fold>

# <editor-fold desc="STORE RESULTS">
date = time.strftime("%d%m%Y-%H%M%S")
results_file_name = (args.results_storage_location +
                     "{}".format(date) + "_" + args.results_id + "RESULTS" + general_script_name +
                     general_storage_file_name + ".pkl")
torch.save({
            "model": model,
            "dataset_info": dataset_info,
            "train_results": train_results,
            "args": args,
            }
           , results_file_name)
print("\nSTORED RESULTS IN FILE:")
print(results_file_name)
# </editor-fold>
