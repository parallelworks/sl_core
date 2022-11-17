#!/bin/bash
#=============================
# Test train_predict_eval.sh
# locally by specifying all
# the necessary inputs with
# data/options that correspond
# to sample files already
# present in this repository.
# This script is a good starting
# point if you want to do
# some initial tests with the
# SuperLearner.
#
#============================

./train_predict_eval.sh ./sample_inputs/whondrs_25_inputs_train.csv \
			25 \
			./sample_inputs/superlearner_conf_sklearn_NNLS.py \
			./sample_outputs/train_predict_eval_output_tmp \
			True \
			False \
			False \
			False \
			4 \
			loky \
			rate.mg.per.L.per.h \
			./sample_inputs/whondrs_25_inputs_predict.csv
