#!/bin/bash
#=======================
# Main workflow launch
# script for sl_core
#
# This script is launched
# when the Execute button
# on the workflow form is
# pressed or if this workflow
# is launched via the PW API.
#
# workflow.xml here defines
# the form and the inputs
# to the workflow.
#======================

#===============================
# Initializaton
#===============================
# Exit if any command fails!
# Sometimes workflow runs fine but there are SSH problems.
# This line is useful for debugging but can be commented out.
#
# In particular, the git operations in this workflow
# **will** fail depending on the situation (i.e. cannot
# clone to existing repo on cluster, or error due to branch
# not already present) -> so for the git portions of the
# workflow to function *as they are* (i.e. without
# substantial error checking/handling additions and/or
# clever usage of git that I have not figured out yet)
# this line **MUST** be commented out.
#set -e

echo " "
echo "===================================="
echo Step 1: Local setup on PW platform
echo Execution is in $0 at `date`
echo " "
echo Workflow parameters from workflow.xml, apirun/main.py, or ./github/workflows/main.yaml:
echo $@
echo " "

# Useful info for context
jobdir=${PWD}
jobnum=$(basename ${PWD})
ssh_options="-o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no"
wfname=sl_core

echo "========================================================="
echo Starting $wfname
echo Running in $jobdir with job number: $jobnum
echo " "

#----------HELPER FUNCTIONS-----------
# Function to read arguments in format "--pname pval" into
# export WFP_<pname>=<pval>.  All varnames are prefixed
# with WFP_ to designate workflow parameter.
f_read_cmd_args(){
    index=1
    args=""
    for arg in $@; do
	prefix=$(echo "${arg}" | cut -c1-2)
	if [[ ${prefix} == '--' ]]; then
	    pname=$(echo $@ | cut -d ' ' -f${index} | sed 's/--//g')
	    pval=$(echo $@ | cut -d ' ' -f$((index + 1)))
	    # To support empty inputs (--a 1 --b --c 3)
	    # Empty inputs are ignored and no env var is assigned.
	    if [ ${pval:0:2} != "--" ]; then
		echo "export WFP_${pname}=${pval}" >> $(dirname $0)/env.sh
		export "WFP_${pname}=${pval}"
	    fi
	fi
	index=$((index+1))
    done
}

# Function to print date alongside with message.
echod() {
    echo $(date): $@
    }

# Convert command line inputs to environment variables.
f_read_cmd_args $@

# List of input arguments converted to environment vars:
echo List of inputs converted to environment variables:
env | grep WFP_
echo " "

# Testing echod
echo "========================================================="
echod Testing echod. Currently on `hostname`.
echod Will excute as $PW_USER@$WFP_whost
echod Running on the following computer: `hostname`
echo " "

# Private key created with ssh-keygen -t ed25519
private_key="/home/${PW_USER}/.ssh/id_ed25519_dynamic-learning-rivers"

# The repository we want to pull, modify, and push back
ml_arch_repo=$WFP_ml_arch_repo

# The branch of the ml_archive repository we want to use
ml_arch_branch=$WFP_ml_arch_branch

# Other repositories we want to include
# TODO: autodetect ML-code repo since running as GH workflow?
ml_code_repo=$WFP_ml_code_repo
ml_data_repo=$WFP_ml_data_repo

# The full path of the location to which the repo will be
# on the remote node.
abs_path_to_arch_repo="/home/${PW_USER}/$(basename $ml_arch_repo)"
abs_path_to_code_repo="/home/${PW_USER}/$(basename $ml_code_repo)"
abs_path_to_data_repo="/home/${PW_USER}/$(basename $ml_data_repo)"

# Name of remote node
remote_node=${WFP_whost}

# Conda environment information
miniconda_loc=$(echo $WFP_miniconda_loc | sed "s/__USER__/${PW_USER}/g")
my_env=$WFP_my_env

# Data paths
work_dir_base=${WFP_work_dir_base}

# Number of instances
# No longer hard coded, option in workflow.xml
#export WFP_num_inst=10

echo Checking inputs to test:
echo user: $PW_USER
echo remote_node: $remote_node
echo private_key: $private_key
echo Checking for private_key: $(ls $private_key)
echo ML archive repo: $ml_arch_repo
echo ML archive branch: $ml_arch_branch
echo ML code repo: $ml_code_repo
echo ML data repo: $ml_data_repo
echo " "
echo "Absolute paths on cluster:"
echo Arch: $abs_path_to_arch_repo
echo Data: $abs_path_to_data_repo
echo Code: $abs_path_to_code_repo
echo " "
echo "Miniconda information:"
echo Location: $miniconda_loc
echo Env. name: $my_env
echo " "
echo "Data flow information:"
echo Working dir basename: $work_dir_base
echo " "
echo "Number of instances"
echo num_inst: $WFP_num_inst
echo "===================================="
echod Step 2: Cluster setup - staging files to head node
echo " "
echo "======> Clone repos to node..."

# See detailed comments for what is happening here in:
# https://github.com/parallelworks/dynamic-learning-rivers/blob/main/test_deploy_key.sh

# ML archive repo must be git cloned with ssh
# b/c using ssh key for auth only if we want to push.
if [ $WFP_push_to_gh = "True" ]; then
    ssh-agent bash -c "ssh-add ${private_key}; ssh -A ${PW_USER}@${remote_node} \"date; git clone ${ml_arch_repo}\""
else
    ssh ${PW_USER}@${remote_node} git clone ${ml_arch_repo}
fi

# Other repos can be pulled via HTTPS or SSH.
ssh ${PW_USER}@${remote_node} git clone ${ml_code_repo}
ssh ${PW_USER}@${remote_node} git clone ${ml_data_repo}

# Force other repos to pull, too.  The clone (above)
# may fail if the repo already exists (i.e. a cluster
# is being used again to run the workflow again). However,
# in that case, it is essential that the repos pull in
# any new updates (because the clone failed, so nothing
# new was pulled).
ssh $PW_USER@$remote_node "cd ${abs_path_to_code_repo}; git pull"
ssh $PW_USER@$remote_node "cd ${abs_path_to_data_repo}; git pull"

echo "======> Create ${ml_arch_branch}..."
ssh $PW_USER@$remote_node "cd ${abs_path_to_arch_repo}; git branch ${ml_arch_branch}"
echo "======> Checkout ${ml_arch_branch}..."
ssh $PW_USER@$remote_node "cd ${abs_path_to_arch_repo}; git checkout ${ml_arch_branch}"

if [ $WFP_push_to_gh = "True" ]; then
    echo "======> Set upstream branch in case branch exists already ${ml_arch_branch}..."
    ssh $PW_USER@$remote_node "cd ${abs_path_to_arch_repo}; git branch --set-upstream-to=origin/${ml_arch_branch} ${ml_arch_branch}"
    ssh-agent bash -c "ssh-add ${private_key}; ssh -A ${PW_USER}@${remote_node} \"cd ${abs_path_to_arch_repo}; git pull\""
fi

echo "======> Test for presence of Conda environment"
ssh $PW_USER@$remote_node "ls /home/$PW_USER/.miniconda*"
if [ $? -ne 0 ]; then
    echo "======> No Conda found; install Conda environment for SuperLearner."
    # Redirect std output from this process
    # to a separate log file.
    echod This process can take several minutes.
    echod See miniconda_install.log for progress.
    ssh $PW_USER@$remote_node "cd ${abs_path_to_code_repo}; ./create_conda_env.sh ${miniconda_loc} ${my_env}" &> miniconda_install.log
else
    echo "======> Conda found!  Assuming no need to install."
fi

# Ensure Conda install is done before proceeding
# This is necessary because stdout and stderr
# in the Conda intall process are redirected
# elsewhere.
wait

echo "===================================="
echod Step 3: Launch jobs on cluster

for (( ii=0; ii<$WFP_num_inst; ii++ ))
do
# Launch a single SuperLearner job
work_dir=${abs_path_to_arch_repo}/${work_dir_base}${ii}
echo "=======> Creating work dir: ${work_dir}"
ssh $PW_USER@$remote_node "mkdir -p ${work_dir}"

echo "======> Launching SuperLearner"
# This launch line can be split over multiple
# lines for readability BUT no spaces are allowed
# outside of " " or the interpreter will assume it's
# the end of the command.
ssh -f ${ssh_options} $PW_USER@$remote_node sbatch" "\
--output=sl.std.out.${remote_node}" "\
--wrap" ""\"cd ${abs_path_to_code_repo}; ./train_predict_eval.sh "\
"${abs_path_to_arch_repo}/${WFP_train_test_data} "\
"${WFP_num_inputs} "\
"${abs_path_to_code_repo}/${WFP_superlearner_conf} "\
"${work_dir} "\
"${miniconda_loc} "\
"${my_env} "\
"${WFP_hpo} "\
"${WFP_cross_val_score} "\
"${WFP_smogn} "\
"${WFP_onnx} "\
"${WFP_n_jobs} "\
"${WFP_backend} "\
"rate.mg.per.L.per.h "\
"${abs_path_to_data_repo}/${WFP_predict_data}""\""
done

echo "===================================="
echod Step 4: Monitor jobs on cluster

# Check if there are any other running jobs on the cluster
# by counting the number of lines in squeue output. One
# line is the header line => no jobs are running.  Anything
# more than 1 means that there is at least one job running.
n_squeue="2"
squeue_wait=10
while [ $n_squeue -gt 1 ]
do
    # Wait first - sbatch launches may take
    # a few seconds to register on squeue!
    echod "Monitor waiting "${squeue_wait}" seconds..."
    sleep $squeue_wait
    n_squeue=$(ssh ${ssh_options} $PW_USER@$remote_node squeue | wc -l )
    echod "Found "${n_squeue}" lines in squeue."
done
echod "No more pending jobs in squeue. Stage SLURM logs back."
rsync $PW_USER@$remote_node:/home/$PW_USER/sl.std.out.${remote_node} ./


echo "===================================="
echod Step 5: Stage files back to GitHub
echo "=====> Add and commit..."
ssh $PW_USER@$remote_node "cd ${abs_path_to_arch_repo}; git add --all ."
ssh $PW_USER@$remote_node "cd ${abs_path_to_arch_repo}; git commit -m \"${jobnum} on $(date)\""

if [ $WFP_push_to_gh = "True" ]; then
    echo "=====> Push..."
    ssh-agent bash -c "ssh-add ${private_key}; ssh -A ${PW_USER}@${remote_node} \"cd ${abs_path_to_arch_repo}; git push origin ${ml_arch_branch}\""
fi

echo "======> Stage files back to PW"
# Although it is duplicating data, this
# step will make it easier to loop over a
# series of job directories for consolidating
# results rather than having to loop through
# git commits.
rsync -av $PW_USER@$remote_node:${abs_path_to_arch_repo}/ml_models ./

echo Done with $0
