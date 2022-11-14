#!/bin/bash
#===========================
# It is essential that the
# workflow parameters (all
# denoted with the "commands|"
# prefix in the JSON string)
# are exactly correct.  Otherwise,
# the workflow will fail without
# any direct error messages to
# the user.
#==========================
${CONDA_PYTHON_EXE} run_workflow.py \
    ${PARSL_CLIENT_HOST} \
    ${PW_API_KEY} \
    ${PW_USER} \
    gcev2 \
    sl_core \
    '{"commands|conda_env": "parsl-pw", "commands|remote_dir": "/tmp", "commands|conda_sh": "/tmp/pworks/.miniconda3/etc/profile.d/conda.sh", "commands|n_jobs": "4", "commands|num_inputs": "9", "commands|cross_val_score": "False", "commands|model_dir": "./model_dir", "commands|hpo": "False", "commands|backend": "threading", "commands|sanity_test": "None"}'
