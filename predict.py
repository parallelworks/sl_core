#================================
# SuperLearner predict script
#================================
# Use a pre-trained SuperLearner
# ML model to make predictions.
#================================

# Dependencies
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import sys
from pprint import pprint

#=======================================
# Main execution
#=======================================
if __name__ == '__main__':

    #===========================
    # Command line inputs
    #===========================
    print("Parsing SuperLearner predict arguments...")
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
    sys.path.append(model_dir)
    # Insert an extra slash just in case missing on command line
    with open(model_dir+"/"+'SuperLearners.pkl','rb') as file_object:
        superlearner = pickle.load(file_object)

    # For a given output variable, list the attributes:
    predict_var = args.predict_var
    print("Submodels within SuperLearner and their weights:")
    print(superlearner[predict_var].named_estimators_.keys())
    print(superlearner[predict_var].final_estimator_.weights_)

print("Done!")
