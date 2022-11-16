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

# Name of remote node
remote_node=${WFP_whost}

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
echo "===================================="
echod Step 2: Cluster setup - staging files to head node
echo " "
echo "======> Clone repos to node..."

# See detailed comments for what is happening here in:
# https://github.com/parallelworks/dynamic-learning-rivers/blob/main/test_deploy_key.sh

# ML archive repo must be git cloned with ssh
# b/c using ssh key for auth.
ssh-agent bash -c "\"ssh-add ${private_key}; ssh -A ${PW_USER}@${remote_node} git clone ${ml_arch_repo}\""

# Other repos can be pulled via HTTPS or SSH.
ssh ${PW_USER}@${remote_node} git clone ${ml_code_repo}
ssh ${PW_USER}@${remote_node} git clone ${ml_data_repo}

echo "======> Create ${ml_arch_branch}..."
ssh $PW_USER@$remote_node "cd ${abs_path_to_arch_repo}; git branch ${ml_arch_branch}"
echo "======> Checkout ${ml_arch_branch}..."
ssh $PW_USER@$remote_node "cd ${abs_path_to_arch_repo}; git checkout ${ml_arch_branch}"
echo "======> Set upstream branch in case branch exists already ${ml_arch_branch}..."
ssh $PW_USER@$remote_node "cd ${abs_path_to_arch_repo}; git branch --set-upstream-to=origin/${ml_arch_branch} ${ml_arch_branch}"
ssh-agent bash -c "ssh-add ${private_key}; ssh -A ${PW_USER}@${remote_node} \"cd ${abs_path_to_arch_repo}; git pull\""

echo "===================================="
echod Step 3: Launch jobs on cluster
echo CURRENTLY JUST WRITING A SIMPLE LOG FILE.
echo INSERT SUPERLEARNER SRUN LAUNCHES HERE

# (Note that this particular repo's .gitignore will ignore filenames
# that match certain patterns, in particular ".log")
ssh $PW_USER@$remote_node "echo Testing on $(date) >> ${abs_path_to_arch_repo}/ml_models/test.std.out"

echo "===================================="
echod Step 4: Monitor jobs on cluster
echo INSERT SQUEUE LISTING CODE FROM SSH_BASH_DEMO

echo "===================================="
echod Step 5: Stage files back to GitHub
echo "=====> Add and commit..."
ssh $PW_USER@$remote_node "cd ${abs_path_to_arch_repo}; git add --all ."
ssh $PW_USER@$remote_node "cd ${abs_path_to_arch_repo}; git commit -m \"Using deploy key on $(date)\""

echo "=====> Push..."
ssh-agent bash -c "ssh-add ${private_key}; ssh -A ${PW_USER}@${remote_node} \"cd ${abs_path_to_arch_repo}; git push origin ${ml_arch_branch}\""

echo Done with $0
