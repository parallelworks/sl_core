#!/bin/bash
#===========================
# Download Miniconda, install,
# and create a new environment
# for using with SuperLearner
# and ONNX.
#
# Eventually, this script
# could be run as part of a
# SL container build.
#
# With a fast internet connection
# (i.e. download time minimal)
# this process takes about 5 min.
#===========================

# Miniconda install location
# The `source` command somehow
# doesn't work with "~", so best
# to put an absolute path here
# if putting Miniconda in $HOME.
miniconda_loc="/home/sfgary/.miniconda3"

# Download current version of
# Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Run Miniconda installer
chmod u+x ./Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p $miniconda_loc

# Clean up
rm ./Miniconda3-latest-Linux-x86_64.sh

# Define environment name
my_env="sl_onnx"

# Define specific versions here
# or leave blank to use whatever
# is automatically selected.
# Currently, sklearn is not compatible
# with Python 3.10, so must use Python 3.9.
python_version="=3.9"
sklearn_version="==0.23.2"
xgboost_version="==1.3.3"
sklopt_version="==0.8.1"

# Start conda
source ${miniconda_loc}/etc/profile.d/conda.sh
conda activate base

# Create new environment
conda create -y --name $my_env python=3.9 ipython

# Jump into new environment
conda activate $my_env

# Install packages
conda install -y pandas
conda install -y matplotlib
conda install -y scikit-learn${sklearn_version}
conda install -y xgboost${xgboost_version}
conda install -y -c conda-forge scikit-optimize${sklopt_version}
conda install -y -c conda-forge onnxmltools
conda install -y -c conda-forge onnxruntime
