#!/bin/bash
#===========================
# Launch several simultaneous runs
# of the SuperLearner
#===========================

echo Start simultaneous_runs.sh

# Specify input file
input_file="../sample_inputs/whondrml_global_train_25_inputs_update.csv"

# Need to create working dirs if they don't exist
workdir1="../sample_outputs/model_test_1"
workdir2="../sample_outputs/model_test_2"

mkdir -p $workdir1
mkdir -p $workdir2

# This simple example launches two runs
./run.sh $input_file $workdir1 &
./run.sh $input_file $workdir2 &

wait

echo Done simultaneous_runs.sh


