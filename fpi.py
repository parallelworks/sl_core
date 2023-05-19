#========================
# SuperLearner FPI
#========================
# Use feature permutation
# importance (FPI) to identify
# which input variables 
# (i.e. features) are the
# most important to a
# trained ML model.
#
# This is a brute-force
# method that basically
# checks the sensitivity
# of ML predictions
# based on selectively 
# mixing up specific
# subsets of features.
#
# Subsets of features are
# permuted together because
# FPI works best if highly
# correlated features are 
# permuted together as a 
# block. Otherwise,
# information encoded a
# single permuted feature 
# could still be avaiable
# to the model via a
# separate correlated feature.
#========================

# Dependencies
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler

#=======================================
# Main execution
#=======================================
if __name__ == '__main__':

    #===========================
    # Command line inputs
    #===========================
    print("Parsing SuperLearner FPI arguments...")
    parser = argparse.ArgumentParser()
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg)
            print(arg)

    args = parser.parse_args()
    
    #===========================================================
    # Load the SuperLearner models
    #===========================================================
    model_dir = args.model_dir
    predict_var = args.predict_var

    sys.path.append(model_dir)
    # Insert an extra slash just in case missing on command line
    with open(model_dir+"/"+'SuperLearners.pkl','rb') as file_object:
        superlearner = pickle.load(file_object)

    # For a given output variable, list the models:
    predict_var = args.predict_var
    print("Submodels within SuperLearner and their weights:")
    list_models = list(superlearner[predict_var].named_estimators_.keys())
    print(list_models)
    
    # The following only works for the scipy.optimize.nnls
    # stacking regressor, not the sklearn stacking regressors.
    print(superlearner[predict_var].final_estimator_.weights_)
    
    #===========================================================
    # Load the training data, predict data, and the actual
    # predictions. It is possible to run FPI on the train, test,
    # and predicted data.
    #===========================================================
    num_inputs = int(args.num_inputs)
    predict_data_csv = args.predict_data+'.csv'
    predict_data_ixy = args.predict_data+'.ixy'
    
    # SuperLearner train.py always saves these files in model_dir:
    train_data = model_dir+'/train.csv'
    test_data = model_dir+'/test.csv'
    
    # We know that the previous SuperLearner steps have
    # also generated the following files:
    predict_output_file = model_dir+"/sl_predictions.csv"
    
    train_df = pd.read_csv(train_data).astype(np.float32)
    X_train = train_df.values[:, :num_inputs]
    Y_train = train_df.values[:, num_inputs:]

    test_df = pd.read_csv(test_data).astype(np.float32)
    X_test = test_df.values[:, :num_inputs]
    Y_test = test_df.values[:, num_inputs:]

    # Lines below may need to be generalized for multi-var
    # predictions.
    X_predict = pd.read_csv(predict_data_csv).astype(np.float32)
    tmp_df = pd.read_csv(predict_output_file).astype(np.float32)
    Y_predict = tmp_df[predict_var]
    
    #===========================================================
    # Find which variables are correlated
    #===========================================================
    
    # Working here
    
    #===========================================================
    # Run FPI
    #===========================================================
    
    # Working here
    
    #===========================================================
    # Make heatmap of variable correlations
    #===========================================================
    
    # Working here
    
    #===========================================================
    # Save outfile file
    #===========================================================
    
    # Working here
    
    #===========================================================
    # Done!
    #===========================================================

