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
    cloud \
    sl_core \
    '{"commands|ml_arch_repo": "https://github.com/parallelworks/dynamic-learning-rivers", "commands|ml_arch_branch": "test-branch", "commands|ml_code_repo": "https://github.com/parallelworks/sl_core", "commands|ml_data_repo": "https://github.com/parallelworks/global-river-databases", "whost": "cloud", "commands|n_jobs": "4", "commands|num_inputs": "9", "commands|cross_val_score": "False", "commands|model_dir": "./model_dir", "commands|hpo": "False", "commands|backend": "threading", "commands|sanity_test": "None"}'
