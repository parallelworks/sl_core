#!/bin/bash
#===========================
# Download Miniconda, install,
# and create a new environment
# for using with SuperLearner
# and ONNX.
#
# Specify the Conda install location
# and environment name, e.g.:
#
# ./create_conda_env.sh ${HOME}/.miniconda3 sl_onnx
#
# Eventually, this script
# could be run as part of a
# SL container build.
#
# With a fast internet connection
# (i.e. download time minimal)
# this process takes < 5 min.
#
# For moving conda envs around,
# it is possible to put the
# miniconda directory in a tarball
# but the paths will need to be
# adjusted.  The download and
# decompression time can be long.
# As an alternative, consider:
# conda list -e > requirements.txt
# to export a list of the req's
# and then:
# conda create --name <env> --file requirements.txt
# to build another env elsewhere.
# This second step runs faster
# than this script because
# Conda does not stop to solve
# the environment.  Rather, it
# just pulls all the listed
# packages assuming everything
# is compatible.
#===========================

echo Starting $0

# Miniconda install location
# The `source` command somehow
# doesn't work with "~", so best
# to put an absolute path here
# if putting Miniconda in $HOME.
#
# Assuming HOME is a universally
# accessible location on the cluster.
miniconda_loc=$1

# Download current version of
# Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Run Miniconda installer
chmod u+x ./Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p $miniconda_loc

# Clean up
rm ./Miniconda3-latest-Linux-x86_64.sh

# Define environment name
my_env=$2

# Define specific versions here
# or leave blank to use whatever
# is automatically selected.
# Currently, sklearn is not compatible
# with Python 3.10, so must use Python 3.9.
# This Python 3.9 works except that one of
# the error message in scikit-learn attempts
# to use decode("latin1") on a string which
# Python 3.9 cannot handle (decode was
# depreciated in Python 3, but I think
# Python 3.7 still supports it somehow
# while Python 3.9 is too far down the
# the road to support decode.)
python_version="=3.7"
sklearn_version="==0.23.2"
xgboost_version="==1.3.3"
sklopt_version="==0.8.1"

# Added here because sklearn 0.23.2 attempts
# to load a thing depreciated from scipy after 1.7.0.
# https://docs.scipy.org/doc/scipy-1.7.1/reference/reference/generated/scipy.linalg.pinv2.html
scipy_version="==1.7.0"

#======================================
# Version adjustment
#======================================
# Try current versions of everything!
python_version="=3.9"
sklearn_version="==1.3.0"
xgboost_version=""
sklopt_version=""
scipy_version=""
# There is an issue with skopt using depreciated np.int, so 
# pin numpy to version that accepts np.int.
numpy_version="==1.22.4"

# Start conda
source ${miniconda_loc}/etc/profile.d/conda.sh
conda activate base

# Create new environment
# (if we are running Jupter notebooks, include ipython here)
conda create -y --name $my_env python${python_version}

# Jump into new environment
conda activate $my_env

# Install packages
conda install -y -c conda-forge scipy${scipy_version}
conda install -y -c conda-forge numpy${numpy_version}
conda install -y pandas
conda install -y matplotlib
conda install -y scikit-learn${sklearn_version}
conda install -y xgboost${xgboost_version}
conda install -y -c conda-forge scikit-optimize${sklopt_version}
conda install -y -c conda-forge onnxmltools
conda install -y -c conda-forge onnxruntime
conda install -y seaborn

# Pip packages last
# SMOGN on pip does not allow for seed option.  Use dev.
#pip install smogn
#pip install git+https://github.com/nickkunz/smogn.git

# conda install -c conda-forge python-igraph installs v0.9.x
# and takes a very long time to solve env.  Perhaps a
# choice of Python issue? Bypass with pip.
pip install igraph==0.10.6

# Force the numpy version to match what we need
conda install -c conda-forge -y numpy${numpy_version}

echo Finished $0
