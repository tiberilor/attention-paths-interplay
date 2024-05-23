import torch
import os
import sys
import numpy as np
import LIB_convergent_summation_heads as lib
import argparse
import time
import logging


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
parser.add_argument("--dataset", default="feb22_1v1_v0_pretrained", type=str,
                    help="[feb22_1v1_v0_pretrained]")
parser.add_argument("--number_examples", "-P", default=100, type=int)
parser.add_argument("--token_readout_style", default="average_pooling", type=str)

# MODEL PARAMETERS
# model size hyperparameters
# parser.add_argument("--projected_input_dimension", "-N0", default=100, type=int) # maybe to implement in the future
parser.add_argument("--model_widths", "-N", nargs='+', type=int, default=[1000],
                    help="single or multiple ints N1 N2 .. NL Na")
parser.add_argument("--variances", nargs='+', type=float, default=[1000],
                    help="single or multiple floats sigma0 sigma1 sigma2 sigmaL sigma_a")
parser.add_argument("--temperature", type=float, default=0.0)

# TRAINING PARAMETERS
parser.add_argument("--learning_rate", "-lr", default=0.001, type=float)
parser.add_argument("--gradient_tolerance", "-tol", default=1e-06, type=float,
                    help="at what size of gradients to stop the training")
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--scheduler_patience", default=1000, type=int,
                    help="number of epochs the scheduler should wait before decreasing the learning rate")
parser.add_argument("--lr_reduce_factor", default=0.1, type=float,
                    help="factor of reduction of the learning rate")
parser.add_argument("--minimum_learning_rate", default=0.001, type=float,
                    help="minimum lr to which the scheduler will reduce the lr")
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
parser.add_argument('--results_file_name', default=None,
                    help='If not None, explicitly specifies path/results file name')
parser.add_argument('--checkpoint_file_name', default=None,
                    help='If not None, explicitly specifies path/results file name')
parser.add_argument('--output_log_file_name', default='./train.log',
                    help='Specifies output path/log file name')
parser.add_argument('--dataset_location', type=str,
                    default="/home/user/datasets",
                    help='where the training dataset is stored')

# OTHER OPTIONS
parser.add_argument("--test_kernel_invertibility", action="store_true",
                    help="This skips the training and just tests the invertibility of the Kernel")
parser.add_argument("--plot_kernel_invertibility", action="store_true",
                    help="Plot the results of testing the invertibility of the Kernel")
parser.add_argument("--test_kernel_invertibility_with_temperature", action="store_true",
                    help="Tests the gp kernel invertibility with the addition of temperature*identity")
parser.add_argument("--hessian_test", action="store_true",
                    help="perform the hessian test to check if we are at a minimum")
parser.add_argument("--auto_tune_lr", action="store_true",
                    help="Automatically set learning rate")

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

# Logging
log_file_name = args.output_log_file_name
handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
logging.basicConfig(
    level=logging.INFO, format='%(message)s', handlers=handlers)

loginf = logging.info
# </editor-fold>

loginf(f"Command executed: {sys.argv[:]}")
loginf(f"\nDEVICE IDENTIFICATION: START")
loginf(f"\nUsing {device} device")
loginf(f"\nDefault precision: {torch.get_default_dtype()}")
loginf(f"\nDEVICE IDENTIFICATION: END")
loginf("\n")

# <editor-fold desc="CONSTRUCT GENERAL STORAGE FILENAME">
# construct the general storage file name
string_list = [str(element) for element in args.variances]
delimiter = "_"
variances_list = delimiter.join(string_list)
string_list = [str(element) for element in args.model_widths]
delimiter = "_"
model_widths_list = delimiter.join(string_list)
general_script_name = "_conv_sum_heads_bin_reg_PRETRAINED_HEADS_theory"
general_storage_file_name = (args.dataset
                             + f"_P{args.number_examples}_"
                               f"N{model_widths_list}_var{variances_list}_"
                               f"lr{args.learning_rate}_upert{args.order_parameter_perturbation_strength}"
                               f"_uscale{args.order_parameter_scale}_"
                               f"useed{args.order_parameter_seed}_"
                               f"sStep{args.number_steps_store_scalars}_tStep{args.number_steps_store_tensors}_")
if args.checkpoint_file_name is None:
    checkpoint_file_name = (args.results_storage_location + "CHECKPOINT" + general_script_name
                            + general_storage_file_name + ".pkl")
else:
    checkpoint_file_name = args.checkpoint_file_name
# </editor-fold>

# START TIME
start_time = time.time()

# <editor-fold desc="INITIALIZE TRAINING DATA">
# initialize the training data
dataset_info = {
    "dataset": args.dataset,
    "number_examples": args.number_examples,
    }

loginf(f'Reading from: {args.dataset_location}')
input, labels, dataset_info = lib.prepare_dataset(args.dataset_location, dataset_info, train=True, loginf=loginf)
loginf(f'done.')
# </editor-fold>

# <editor-fold desc="INITIALIZE MODEL">
# initialize the model...
# ...from scratch
if not os.path.isfile(checkpoint_file_name):
    loginf('No checkpoint')
    # if no checkpoint is present, the starting epoch should be 0
    starting_epoch = 0

    # retrieve some info to intialize the model
    input_width = input.size()[1]
    if type(dataset_info["number_heads"]) is not list:
        numbers_heads = [dataset_info["number_heads"]]  # this should be a list
    else:
        numbers_heads = dataset_info["number_heads"]
    qk_internal_dimension = dataset_info["qk_internal_dimension"]
    number_attention_layers = dataset_info["number_attention_layers"]

    loginf('Define model...')
    model = lib.ConvergentSummationHeads(numbers_heads, args.model_widths, number_attention_layers,
                                         input_width, args.variances, token_readout_style=args.token_readout_style,
                                         temperature=args.temperature)
    loginf('done.')
    # we use the "first_token" readout style, as this is the configuration we had when we pretrained the attention heads
    # set attention weights
    model.set_qk(dataset_info["query_weights"], dataset_info["key_weights"])
    # delete this info from dataset info (to free some memory, as this information is now redundant)
    dataset_info["query_weights"] = None
    dataset_info["key_weights"] = None
    # initialize order parameters
    model.set_order_parameters_gp_perturbed(perturbation_strength=args.order_parameter_perturbation_strength,
                                            seed=args.order_parameter_seed, scale=args.order_parameter_scale)
# ...from checkpoint
else:
    loginf(f'Load checkpoint from {checkpoint_file_name}')
    checkpoint = torch.load(checkpoint_file_name)
    # retrieve the model
    model = checkpoint["model"]
    # retrieve the current training_results
    train_results = checkpoint["train_results"]
    # set the starting epoch from where the checkpoint was saved
    starting_epoch = checkpoint["reached_epoch"] + 1
    loginf(f"\nFOUND A CHECKPOINT.\nStarting from epoch {starting_epoch}")
    loginf("\n")
    loginf("MODEL INFO: START")
    loginf("\n")
    model.print_architecture()
    loginf("\n")
    loginf("MODEL INFO: END")
    loginf("\n")

# </editor-fold>

# <editor-fold desc="SETUP OPTIMIZER, MODEL, DATASET FOR TRAINING">

labels = labels.to(device)

if args.auto_tune_lr:
    loginf('Autotune learning rate...')
    # check for 5 epochs
    min_energy = np.inf
    best_lr = 0.
    for cur_lr in [1.0, 0.8, 0.5, 0.1, 0.08, 0.05, 0.01, 0.008, 0.005, 0.001, 0.0008, 0.0005, 0.0001]:
        loginf(f"current lr: {cur_lr}")
        loginf(f"set...")
        model.set_order_parameters_gp_perturbed(
            perturbation_strength=args.order_parameter_perturbation_strength,
            seed=args.order_parameter_seed, scale=args.order_parameter_scale)
        loginf(f"load...")
        model.load(input, dataset_info)
        loginf(f"start")
        model.to_device(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cur_lr)
        energy_values = []
        nan_break = False
        for iter in range(10):
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

            total_loss = energy + entropy
            optimizer.step()
            energy_values.append(energy.item())

            max_gradients = []
            for l, param in enumerate(model.parameters()):
                max_abs_grad = torch.max(torch.abs(param.grad)).detach().cpu().clone().numpy()
                max_gradients.append(max_abs_grad)
            max_grad = np.max(max_gradients)
            loginf(f'[lr {cur_lr}, iter {iter}] enery: {energy:.2f}, entropy: {entropy:.2f}, '
                   f'total: {total_loss:.2f}, max_grad: {max_grad:.2f}')

            if np.isnan(entropy.item()):
                loginf(f"[lr {cur_lr}, iter {iter}] Nan entropy, stopping training.")
                nan_break = True
                break
            elif entropy.item() < 0:
                loginf(f"[lr {cur_lr}, iter {iter}] Negative entropy, stopping training.")
                nan_break = True
                break
            if np.isnan(energy.item()):
                loginf(f"[lr {cur_lr}, iter {iter}] Nan energy, stopping training.")
                nan_break = True
                break

        if nan_break:
            mean_energy = np.inf
        else:
            mean_energy = np.mean(energy_values)
        if mean_energy < min_energy:
            loginf(f'Best error sofar, from {min_energy:.2f} to {mean_energy:.2f}')
            loginf(f'update best lr, from {best_lr} to {cur_lr}')
            min_energy = mean_energy
            best_lr = cur_lr
        else:
            loginf(f'Not better')
    loginf(f'Use best lr: {best_lr}')
    model.set_order_parameters_gp_perturbed(
        perturbation_strength=args.order_parameter_perturbation_strength,
        seed=args.order_parameter_seed, scale=args.order_parameter_scale)

# load the model for training
model.load(input, dataset_info)

# send everything to device
model.to_device(device)  # this is a special function created for the model to map all that is needed to a device

# initialize the optimizer
if not os.path.isfile(checkpoint_file_name):
    if args.auto_tune_lr:
        optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
        last_lr = best_lr
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        last_lr = args.learning_rate
else:
    checkpoint = torch.load(checkpoint_file_name)
    if "optimizer" in checkpoint_file_name:
        optimizer = checkpoint["optimizer"]
        if args.auto_tune_lr:
            last_lr = best_lr
        else:
            last_lr = args.learning_rate
    else:
        if args.auto_tune_lr:
            optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
            last_lr = best_lr
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            last_lr = args.learning_rate


# initialize the learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_reduce_factor,
                                                       patience=args.scheduler_patience,
                                                       min_lr=args.minimum_learning_rate)
# </editor-fold>

# <editor-fold desc="test invertibility of GP Kernel (optional, exits the script afterward)">
if args.test_kernel_invertibility:
    model.test_gp_kernel_invertibility(plot=args.plot_kernel_invertibility,
                                       with_temperature=args.test_kernel_invertibility_with_temperature)
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
        loginf(f"Achieved required tolerance, terminating training at step {t}")
        achieved_tolerance = True
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
        loginf(f"[epoch {t+1}] Saving checkpoint to: {checkpoint_file_name}")
        # unload the model, so we do not store useless data
        attentioned_input = model.unload_before_checkpoint()
        # save
        torch.save({
            "reached_epoch": t,
            "model": model,
            "optimizer": optimizer.state_dict(),
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
        loginf(f"\n[{time.strftime('%Y/%m/%d--%H:%M:%S')}] Loss at step {t}/{args.epochs}")
        loginf(f"Total: {loss.item():>7f} (Energy: {energy.item():>7f}; Entropy: {entropy.item():>7f})")
        loginf(f"max abs gradient: {max_grad}")

    if np.isnan(max_grad):
        loginf("Nan max gradient, stopping training.")
        break
    if np.isnan(entropy.item()):
        loginf(f"Nan entropy: {entropy.item()}, stopping training.")
        break
    if entropy.item() < 0:
        loginf(f"Negative entropy: {entropy.item()}, stopping training.")
        break
    if np.isnan(energy.item()):
        loginf(f"Nan energy: {energy.item()}, stopping training.")
        break

    # exit training if we achieved tolerance
    if achieved_tolerance:
        break

    # do one optimization step(unless we are at the last step)
    if t != (args.epochs - 1):
        optimizer.step()

    # do one step of the learning_rate scheduler
    scheduler.step(max_grad)
    current_lr = scheduler.get_last_lr()[0]
    if current_lr != last_lr:
        loginf(f"Updated lr from {last_lr} to {current_lr}")
    last_lr = current_lr

assert achieved_tolerance, 'Do not accept training end without convergence'

# </editor-fold>

# <editor-fold desc="PRINT END-OF-TRAINING INFO">
loginf("\nEND-OF-TRAINING-INFO: START\n")
# print loss info
loginf(f"Final loss at step {train_results['time_points_history'][-1]}/{args.epochs}:")
loginf(f"Total: {train_results['loss_history'][-1]:>7f} (Energy: {train_results['loss_energy_history'][-1]:>7f}; "
      f"Entropy: {train_results['loss_entropy_history'][-1]:>7f})")
# print max gradient info
max_gradients = []
for l, param in enumerate(model.parameters()):
    max_abs_grad = torch.max(torch.abs(param.grad)).detach().cpu().clone().numpy()
    max_gradients.append(max_abs_grad)
max_grad = np.max(max_gradients)
loginf(f"max abs gradient: {max_grad}")
# print final order parameter info
final_order_param = model.order_parameters[-1].detach().cpu().clone()
size = final_order_param.size()[0]
off_diagonals = final_order_param[torch.triu(torch.ones(size, size)) == 1]
diagonals = torch.diag(final_order_param)
mean_off_diagonals = torch.mean(torch.abs(off_diagonals))
std_off_diagonals = torch.std(torch.abs(off_diagonals))
mean_diagonals = torch.mean(torch.abs(diagonals))
std_diagonals = torch.std(torch.abs(diagonals))
loginf(f"ORDER PARAMETER INFO:")
loginf(f"mean(abs) diagonal: {mean_diagonals}")
loginf(f"std(abs) diagonal: {std_diagonals}")
loginf(f"mean(abs) off-diagonal: {mean_off_diagonals}")
loginf(f"std(abs) off-diagonal: {std_off_diagonals}")
loginf("\nEND-OF-TRAINING-INFO: END\n")
# </editor-fold>

# <editor-fold desc="CHECK THE HESSIAN (optional)">
if args.hessian_test:
    model.perform_hessian_test(labels)
# </editor-fold>

# unload the model after training
model.unload()

# <editor-fold desc="PRINT TIME AND MEMORY USAGE INFO">
loginf(f"\nSCRIPT PERFORMANCE INFO: START")
max_gpu_memory_allocated = torch.cuda.max_memory_allocated(device="cuda")
loginf("\nMaximum GPU memory allocated:")
loginf(f"GPU: {max_gpu_memory_allocated/10**9} GB ({max_gpu_memory_allocated/10**6} MB)")
time_elapsed = time.time() - start_time
loginf("\ntotal running time (mins):")
loginf(time_elapsed / 60)
loginf("\ntotal running time (hrs):")
loginf(time_elapsed / 3600)
loginf(f"\nSCRIPT PERFORMANCE INFO: END")
loginf("\n")
# </editor-fold>

# <editor-fold desc="STORE RESULTS">
date = time.strftime("%d%m%Y-%H%M")
if args.results_file_name is None:
    results_file_name = (args.results_storage_location +
                        "{}_RESULTS".format(date) + general_script_name +
                        general_storage_file_name + f"_e{args.epochs}" + ".pkl")
else:
    results_file_name = args.results_file_name
torch.save({
            "model": model,
            "dataset_info": dataset_info,
            "train_results": train_results,
            "args": args,
            }
           , results_file_name)
# </editor-fold>
