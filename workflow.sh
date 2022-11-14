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

echo Starting test workflow at `date`

echo INPUT ARGUMENTS:
echo $@

echo Running on the following computer: `hostname`

echo This is a NO-OP workflow - nothing got launched remotely.

echo Done!
