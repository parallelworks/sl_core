#================================
# SuperLearner PCA script
#================================
# Use PCA analysis to determine
# which data points are the most
# or least similar and use that
# information, along with the
# error estimate, to rank which
# data points are the most
# "important" for training.
#================================

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
import sys
from pprint import pprint

#=======================================
# Main execution
#=======================================
if __name__ == '__main__':

    #===========================
    # Command line inputs
    #===========================
    print("Parsing SuperLearner PCA arguments...")
    parser = argparse.ArgumentParser()
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg)
            print(arg)

    args = parser.parse_args()

    #===========================================================
    # Load the training data, predict data, and the actual
    # predictions. The actual predictions are only loaded here
    # so that they can be easily merged with the final ranking.
    # Some data points are lost along the way due to missing
    # values (flagged with NaN) and we don't want those points
    # to end up biasing the PCA.
    #===========================================================
    num_inputs = int(args.num_inputs)
    model_dir = args.model_dir
    predict_data = model_dir+'/'+args.predict_data
    predict_output = model_dir+'/'+args.predict_output
    pca_output = model_dir+"sl_pca.csv"

print("Done!")
