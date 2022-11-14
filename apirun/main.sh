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
    ssh_bash_demo \
    '{"commands|whost": "gcev2.clusters.pw", "commands|rundir": "/var", "commands|runcmd": "ls -a -l -h", "commands|spaces_in_runcmd": "True", "commands|sleep_time": "10", "commands|head_or_worker": "False"}'
