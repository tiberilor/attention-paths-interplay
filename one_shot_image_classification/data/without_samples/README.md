## Code/without_samples

To produce the data:
1. prodcue the data scanning over a range of temperatures, following the instructions in the ./all_temperatures directory. Then run "RUN_select_optimal_temperature.sh" to select only the datafiles corresponding to the optimal temperature for a given N
2. Run "RUN_sample_posterior.sh" on each of the prodcued data files. This will produce a new copy of the data file, which contains in addition the sampled weights. This script may need to be run on a cluster, possibly on GPU.
3. Move the obtained data files to "../"

