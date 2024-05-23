import argparse
import LIB_convergent_summation_heads as lib
import torch
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', "-f", type=str)
parser.add_argument('--sampling_device', type=str, default="gpu", help="gpu or cpu")
parser.add_argument("--minimum_sampling_temperature", type=float, default=1.0e-05)
parser.add_argument("--seed", type=int)
parser.add_argument("--number_warmup", type=int, default=100)
parser.add_argument("--number_samples", type=int, default=100)
parser.add_argument("--number_skip_samples", type=int, default=1)
parser.add_argument("--number_chains", type=int, default=2)
parser.add_argument("--overwrite_stored_samples", action="store_true",
                    help='If this flag is used, the new samples will overwrite previously stored ones, if present. '
                         'Otherwise, new samples are concatenated to previously stored ones')
parser.add_argument('--dataset_location', type=str,
                    default="./",
                    help='where the training datasets are stored')

args = parser.parse_args()

# <editor-fold desc="LOAD MODEL">
results = torch.load(args.file_name, map_location=torch.device('cpu'))
# we convert what we loaded to cpu, so we make sure that even if we saved things on gpu, they are still properly
# loaded
dataset_info_train = results["dataset_info"]
model = results["model"]
old_args = results["args"]
# </editor-fold>

# <editor-fold desc="FORCE FLAGS">
if old_args.single_precision:
    torch.set_default_dtype(torch.float32)
    print("single precision")
else:
    torch.set_default_dtype(torch.float64)
    print("double precision")
# </editor-fold>

# <editor-fold desc="ENSURE BACKWARDS COMPATIBILITY">
# for compatibility with older results which do not have the sampling-related members:
if not hasattr(model, 'posterior_samples'):
    # If 'posterior_samples' attribute does not exist, add it
    setattr(model, 'posterior_samples', None)
    # add also another attribute posterior_sampling_info:
    posterior_sampling_info = {
        "number_runs": 0,  # the number of different times the posterior has been sampled, and the results appended
        "number_chains": [],
        "number_warmups": [],
        "number_samples_per_chain": [],
        "tot_number_samples": [],
        "divergences": [],  # list of list, for each run, it is a list of the divergences at different chains
        "BFMIs": [],  # list of list, as above
        "avg_acceptance_probabilities": [],  # list of list, as above
        "seeds": []
    }
    setattr(model, 'posterior_sampling_info', posterior_sampling_info)
elif "number_warmups" not in model.posterior_sampling_info:
    model.posterior_sampling_info["number_warmups"] = []
# </editor-fold>

# <editor-fold desc="SAMPLE">
# retrieve train dataset
train_input, train_labels, dataset_info_sample = lib.prepare_dataset(None, dataset_info_train,
                                                                     train=True)
# set minimum temperature for sampling
model.set_minimum_temperature_posterior_sampling(args.minimum_sampling_temperature)
# do inference
model.sample_bayesian_posterior(args.seed, train_input, train_labels, args.number_warmup, args.number_samples,
                                args.number_chains, device=args.sampling_device,
                                overwrite=args.overwrite_stored_samples, number_skip_samples=args.number_skip_samples)
# </editor-fold>

# <editor-fold desc="STORE RESULTS">
if "PLUS_SAMPLES" in args.file_name:
    old_filename = args.file_name.split("_PLUS_SAMPLES")[0]
else:
    old_filename = args.file_name.rsplit(".", 1)[0]
date = time.strftime("%d%m%Y-%H%M%S")
new_attachment = "_PLUS_SAMPLES_" + "{}_".format(date) + model.return_posterior_sampling_filename() + ".pkl"
results_file_name = old_filename + new_attachment
# check if filename exceeds max length:
if len(os.path.basename(results_file_name)) > 255:
    print("\033[91m Warning: filename length exceed maximum size, storing file with name \033[0m")
    old_filename = args.file_name.rsplit(".", 1)[0]
    results_file_name = old_filename + "NEW" + "_{}_".format(date) + ".pkl"
    print("\033[91m Warning: filename length exceed maximum size, storing file with this name instead: \033[0m")
    print(results_file_name)
if len(os.path.basename(results_file_name)) > 255:
    old_filename = args.file_name.rsplit(".", 1)[0]
    results_file_name = old_filename + "_NEW" + ".pkl"
    print("\033[91m Warning: filename length still exceed maximum size, storing file with this name instead: \033[0m")
    print(results_file_name)
if len(os.path.basename(results_file_name)) > 255:
    old_filename = args.file_name.rsplit(".", 1)[0]
    results_file_name = old_filename + "N" + ".pkl"
    print("\033[91m Warning: filename length still exceed maximum size, storing file with this name instead: \033[0m")
    print(results_file_name)
if len(os.path.basename(results_file_name)) > 255:
    results_file_name = args.file_name
    print("\033[91m Warning: filename length still exceed maximum size. "
          "THE ORIGINAL FILE HAS BEEN OVERWRITTEN \033[0m")

torch.save({
            "model": model,
            "dataset_info": results["dataset_info"],
            "train_results": results["train_results"],
            "args": results["args"],
            "args_sampling": args,
            }
           , results_file_name)

print("\nSTORED RESULTS IN FILE:")
print(results_file_name)
# </editor-fold>
