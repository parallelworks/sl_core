#!/bin/bash
#===========================
# Download Miniconda, install,
# and create a tinyu new environment
# for simple testing.
#
# Specify the Conda install location
# and environment name, e.g.:
#
# ./create_conda_env.sh ${HOME}/.miniconda3 tiny
#
# With a fast internet connection
# (i.e. download time minimal)
# this process takes < 5 min.
#
# It is possible to put a
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
#
# Another alternative is to export
# a Conda env to a .yaml file:
# conda env export --name my_env > requirements.yaml
# and then build an environment from
# this .yaml:
# conda env update --name my_new_env -f requirements.yaml
#
# Using a .txt or .yaml is often
# much faster than explicitly listing
# several conda install commands in
# a script. This is because the
# dependencies are already resolved
# in the .txt and .yaml files (i.e.
# they come from already working
# Conda envs) whereas the solving
# process needs to be started over
# from scratch when explicitly listing
# conda install commands.
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

#======================================
# Version adjustment
#======================================
# Try current versions of everything!
python_version=""

# Start conda
source ${miniconda_loc}/etc/profile.d/conda.sh
conda activate base

# Create new environment
# (if we are running Jupter notebooks, include ipython here)
conda create -y --name $my_env python${python_version}

# Jump into new environment
conda activate $my_env

# Install packages
conda install -y -c conda-forge dask
conda install -y -c conda-forge distributed

# Pip packages last

# Force the numpy version to match what we need
conda install -c conda-forge -y numpy${numpy_version}

echo Finished $0
