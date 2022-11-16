#!/bin/bash
#======================
# For a complicated workflow
# like the SuperLearner with
# many workflow parameters
# defined in the workflow.xml,
# it can be challenging to
# detect issues in the API
# workflow-params payload
# (with all the commands|)
# and the required commands
# in the form.
#
# Any discrepancy between what
# the workflow.xml defines as
# inputs and the API payload
# will result in the workflow
# launch using the default values
# in workflow.xml **WITHOUT
# POSTING ANY ERRORS TO THE USER**
# so it can be a challenge to
# define complicated API payloads.
# While error messages to support
# API launched workflow development
# are ongoing work, running this
# script is a quick alternative.
#
# Here, we get all the workflow
# parameters listed in workflow.xml
# and the workflow parameters in
# ./apirun/main.sh to compare them
# side by side to check the number
# of inputs, spelling, etc.
#=================================

api_launcher=main.sh
xml_launcher=../workflow.xml
yml_launcher=../../dynamic-learning-rivers/.github/workflows/main.yml

echo '===================================='
echo Checking $api_launcher
echo '===================================='
grep commands $api_launcher | sed 's/,/\n/g'
echo ' '

echo '===================================='
echo Checking $xml_launcher
echo '===================================='
grep param.name $xml_launcher
echo ' '

echo '===================================='
echo Checking $yml_launcher
echo '===================================='
grep commands $yml_launcher | sed 's/,/\n/g'
echo ' '
