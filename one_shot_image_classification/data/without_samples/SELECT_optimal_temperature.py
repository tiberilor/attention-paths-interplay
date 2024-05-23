import argparse
import LIB_convergent_summation_heads as lib
import torch
import copy
import shutil
import os


parser = argparse.ArgumentParser()
parser.add_argument('--filenames', nargs='+')

parser.add_argument("--number_test_examples", default=100, type=int)

parser.add_argument('--dataset_location', type=str,
                    default="./",
                    help='where the training dataset is stored')
args = parser.parse_args()

number_test_examples = args.number_test_examples

# FORCE FLOAT64
torch.set_default_dtype(torch.float64)

# I wrap everything in torch.no_grad(),
# so that graphs are not accumulated through the loop over the many different files!

with torch.no_grad():
    # Initialize a dictionary to store filenames for each width value
    width_file_mapping = {}

    # Iterate through each filename
    for filename in args.filenames:
        # Load data from the file (assuming it contains the width value)
        results = torch.load(filename, map_location=torch.device('cpu'))
        width = results["model"].model_widths[0]
        # width = results["args"].model_widths[0]  # this was old, but better not to read from args, which may change

        # Check if the width value is already a key in the dictionary
        if width not in width_file_mapping:
            # If not, create a new list for this width value
            width_file_mapping[width] = []

        # Append the current filename to the list corresponding to the width value
        width_file_mapping[width].append(filename)

    # Now width_file_mapping contains lists of filenames for each width value
    # Each key corresponds to a different a width value

    # LOOP THROUGH THE DIFFERENT MODEL WIDTHS
    filenames_optimal_temperature = []
    for width in width_file_mapping.keys():

        filenames = width_file_mapping[width]
        print(f"model width: {width}")

        # initialize a dummy current accuracy
        accuracy_current = 0.0
        # initialize current filename
        filename_current = None

        # loop through different temperatures, find the optimal one
        for filename in filenames:
            # load results
            results = torch.load(filename, map_location=torch.device('cpu'))
            # we convert what we loaded to cpu, so we make sure that even if we saved things on gpu, they are still properly
            # loaded
            dataset_info_train = copy.deepcopy(results["dataset_info"])

            model = results["model"]
            # args_simulation = results["args"]
            # temperature = args_simulation.temperature  # better read from model, rather than args, as done below
            temperature = model.temperature
            print(f"temperature: {temperature}")

            # LOAD THE MODEL FOR TESTING
            train_input, train_labels, dataset_info_train = lib.prepare_dataset(args.dataset_location, dataset_info_train,
                                                                                train=True)
            # load the model for testing
            model.load(train_input, dataset_info_train)

            # <editor-fold desc="TEST PREDICTOR (IN DISTRIBUTION)">
            # retrieve test data
            dataset_info_test = copy.deepcopy(results["dataset_info"])
            dataset_info_test["number_examples"] = number_test_examples

            test_input, test_labels, _ = lib.prepare_dataset(args.dataset_location, dataset_info_test, train=False)

            # compute predictor statistics (renormalized)
            predictor_mean, _ = model.compute_predictor_statistics(test_input, train_labels, gp_limit=False)
            thresholded_predictor_mean = predictor_mean / torch.abs(predictor_mean)
            # add to test_labels (result will be +2 or -2 for correct classification, 0 otherwise). Take abs, divide by 2
            # and sum: the sum will be the number of correctly classified examples. (so taking the mean gives the accuracy in %)
            accuracy_mean_predictor_renormalized = torch.mean(torch.abs(thresholded_predictor_mean + test_labels)/2)

            # </editor-fold>

            # check whether accuracy has improved. Update the current best filename
            if accuracy_mean_predictor_renormalized > accuracy_current:
                accuracy_current = accuracy_mean_predictor_renormalized
                filename_current = filename

        # append the identified filename at optimal temperature
        filenames_optimal_temperature.append(filename_current)

    # copy files corresponding to optimal temperatures
    for filename in filenames_optimal_temperature:
        # Split the filename into folders and the actual filename
        folders, file_name = filename.rsplit('/', 1)
        folders = folders.rsplit('/', 1)[0]  # Remove the last folder
        new_filename = folders + '/' + file_name + ".pkl"

        # Copy the original file to the new location
        shutil.copy(filename, new_filename)
