#!/bin/bash
#=============================
# Test train_predict_eval.sh
# locally by specifying all
# the necessary inputs with
# data/options that correspond
# to sample files already
# present in this repository.
# This script is a good starting
# point if you want to do
# some initial tests with the
# SuperLearner.
#
#============================

# One place to specify Conda install
# location for the rest of the pipeline.
miniconda_loc="${HOME}/.miniconda3"
my_env="superlearner"

# Specify the output (working) directory
# and make it if it does not yet exist.
work_dir="./sample_outputs/train_predict_eval_output_tmp"
mkdir -p $work_dir

echo Starting $0

# Check if Conda environment is installed.
echo "======> Test for presence of Conda environment"
ls $miniconda_loc > /dev/null
if [ $? -ne 0 ]; then
    echo "======> No Conda found; install Conda environment for SuperLearner."
    ./create_conda_env.sh $miniconda_loc $my_env
else
    echo "======> Conda found!  Assuming no need to install."
fi

./train_predict_eval.sh \
    ./sample_inputs/whondrs_25_inputs_train.csv \
    25 \
    ./sample_inputs/superlearner_conf.py \
    $work_dir \
    $miniconda_loc \
    $my_env \
    True \
    True \
    True \
    False \
    4 \
    loky \
    Respiration_Rate_mg_per_L_per_H \
    ./sample_inputs/whondrs_25_inputs_predict

echo Finished $0
