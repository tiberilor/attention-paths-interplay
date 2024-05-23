#!/bin/bash

source /.../anaconda3/bin/activate montecarlo_test
python SELECT_optimal_temperature.py \
--filenames ./all_temperatures/* \
--number_test_examples 1000 \

