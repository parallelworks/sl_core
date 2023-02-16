#=====================================
# SuperLearner launch script
#=====================================
# The user must specify the training
# data set and the working directory,
# e.g.
#
# run.sh /path/to/data.csv /another/path/to/model_dir
#
#====================================

#====================================
# Command line requirements
#====================================
# Define the training data (input)
input_data=$1

# WORNKING HERE! WILL NEED TO INCLUDE NUM_INPUTS!

# Define the work dir (where to run and put output)
work_dir=$2

#===================================
# Conda activate and log env
#===================================
source $HOME/mambaforge/etc/profile.d/mamba.sh
conda activate sl_onnx
mamba list -e > ${work_dir}/requirements.txt

#===================================
# Run the SuperLearner
#===================================
python -m main \
       --conda_sh '~/mambaforge/etc/profile.d/conda.sh' \
       --superlearner_conf '../sample_inputs/superlearner_conf_NNLS_TTR.py' \
       --n_jobs '8' \
       --num_inputs '25' \
       --cross_val_score 'True' \
       --model_dir ${work_dir} \
       --hpo 'True' \
       --data ${input_data} \
       --backend 'loky' 1> ${work_dir}/std.out 2> ${work_dir}/std.err

#===================================
# Print out information about the
# model and make predictions
#===================================

# WORKING HERE

#===================================
# Run PCA on predictions
#===================================

# WORKING HERE

