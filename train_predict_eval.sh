#!/bin/bash
#=====================================
# SuperLearner launch script
#=====================================
# The user must specify:
# 1) the training/testing data set,
# 2) the number of inputs
# 3) the SuperLearner configuration,
# 4) the working directory,
# 5) Boolean flags (HPO, CV, SMOGN, ONNX) 
# For example:
#
# train_predict_eval.sh /path/to/data.csv \  #-----Core params------
#               $NUM_INPUTS
#               /path/to/sl_conf.py \
#               /path/to/work_dir \
#               /path/to/conda_install
#               $CONDA_ENV_NAME
#               $HPO_true_or_false \         #-----Bool opts---------
#               $CV_true_or_false \
#               $SMOGN_true_or_false \
#               $ONNX_true_or_false \
#               $NUM_JOBS \                  #-----HPC opts----------
#               $BACKEND \
#               $PREDICT_VAR \               #-----Predict opts------
#               /path/to/predict_data
#====================================

echo Starting $0
echo The option for ONNX is ignored!

#====================================
# Command line requirements
#====================================
# Define the training data (input)
input_data=$1

# Set the number of inputs
num_inputs=$2

# Set the SuperLearner configuration
sl_conf=$3

# Define the work dir (where to run and put output)
work_dir=$4

# Define Conda environment location and name
miniconda_loc=$5
my_env=$6

# Workflow boolean options (all either True or False)
hpo=$7
cv=$8
smogn=$9
onnx=${10}

# HPC options
num_jobs=${11}
backend=${12}

# Predict options
predict_var=${13}
predict_data=${14}

echo Checking command line inputs:
echo input_data $input_data
echo num_inputs $num_inputs
echo sl_conf $sl_conf
echo work_dir $work_dir
echo miniconda_loc $miniconda_loc
echo my_env $my_env
echo hpo $hpo
echo cv $cv
echo smogn $smogn
echo onnx $onnx
echo num_jobs $num_jobs
echo backend $backend
echo predict_var $predict_var
echo predict_data $predict_data

#===================================
# Conda activate and log env
#===================================
source ${miniconda_loc}/etc/profile.d/conda.sh
conda activate $my_env
# Save Conda env setup and zip the file
# because we don't want GH Dependabot to
# interpret it as the repo's actual
# requirements since this environment is
# emphemeral (otherwise, GH may detect
# security risks in packages and send lots
# of warnings). The HPC code in this Conda
# environment is being executed entirely
# within the environment of the cluster and
# is not exposed to the outside world.
conda list -e | gzip -1c > ${work_dir}/requirements.txt.gz

#===================================
# Run the SuperLearner
#===================================
python -m train \
       --conda_sh "${miniconda_loc}/etc/profile.d/conda.sh" \
       --superlearner_conf $sl_conf \
       --n_jobs $num_jobs \
       --num_inputs $num_inputs \
       --predict_var ${predict_var} \
       --cross_val_score $cv \
       --model_dir ${work_dir} \
       --hpo $hpo \
       --smogn $smogn \
       --data ${input_data} \
       --backend $backend 1> ${work_dir}/train.std.out 2> ${work_dir}/train.std.err

#===================================
# Print out information about the
# model and make predictions
#===================================

python -m predict \
       --model_dir ${work_dir} \
       --predict_var ${predict_var} \
       --num_inputs 25 \
       --predict_data ${predict_data} 1> ${work_dir}/predict.std.out 2> ${work_dir}/predict.std.err

#===================================
# Run PCA on predictions
#===================================

python -m pca \
       --model_dir ${work_dir} \
       --num_inputs 25 \
       --data ${input_data} \
       --predict_var ${predict_var} \
       --predict_data ${predict_data} 1> ${work_dir}/pca.std.out 2> ${work_dir}/pca.std.err

#===================================
# Run FPI
#===================================

python -m fpi \
       --model_dir ${work_dir} \
       --predict_var ${predict_var} \
       --num_inputs 25 \
       --predict_data ${predict_data} 1> ${work_dir}/predict.std.out 2> ${work_dir}/predict.std.err

#===================================
# Compress outputs
#===================================

cd $work_dir
ls
#WORKING HERE

#===================================
echo $0 finished!
