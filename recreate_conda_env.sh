#!/bin/bash
#=============================
# Creating a Conda env from
# scratch each time takes
# several minutes and can fail
# when one dependency depreciates
# functionality and the other
# dependencies haven't adapted
# to the new release. So,
# instead select the specification
# of a good environment and
# always reconstruct based on
# that environment.
#==============================

echo Starting $0

# Miniconda install location
# The `source` command somehow
# doesn't work with "~", so best
# to put an absolute path here
# if putting Miniconda in $HOME.
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

# Define environment spec
my_spec=$3

# Start conda
source ${miniconda_loc}/etc/profile.d/conda.sh
conda activate base

# Create the new environment
conda create -c conda-forge --name $my_env --file $my_spec

#Done

