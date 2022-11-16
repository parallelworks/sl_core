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
    '{"commands|ml_arch_repo": "git@github.com:parallelworks/dynamic-learning-rivers-ha", "commands|ml_arch_branch": "test-branch", "commands|ml_code_repo": "https://github.com/parallelworks/sl_core", "commands|ml_data_repo": "https://github.com/parallelworks/global-river-databases", "commands|whost": "cloud", "commands|train_test_data": "empty.csv", "commands|predict_data": "empty.csv", "commands|num_inputs": "25", "commands|superlearner_conf": "empty.py", "commands|hpo": "True", "commands|cross_val_score": "False", "commands|smogn": "False", "onnx": "False", "commands|backend": "loky", "commands|n_jobs": "4", "commands|model_dir": "./model_dir"}'
