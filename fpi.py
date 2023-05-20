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
    
    #==========================================================
    # FPI functions
    #==========================================================
    
    def permute_importance(
        permutation_feature_blocks_str, model, X, y, 
        scoring_func, n_repeats=20, ratio_score=True):

        base_preds = model.predict(X.values)
        base_score = scoring_func(y, base_preds)

        blocks, block_names = parse_permutation_feature_blocks(
            permutation_feature_blocks_str, X.columns)
        block_scores = list()
        for block, block_name in zip(blocks, block_names):
            block_df = X.copy()
            repeat_scores = list()
            for i_repeat in range(n_repeats):
                print('For block '+block[0]+' iteration '+str(i_repeat))
                block_df[block] = shuffle(block_df[block]).values
                repeat_preds = model.predict(block_df.values)
                repeat_scores.append(scoring_func(y, repeat_preds))

            # Get block score
            importance_score_mean = np.mean(repeat_scores) / base_score if ratio_score else np.mean(repeat_scores) - base_score
            importance_score_std = np.std(repeat_scores) / base_score if ratio_score else np.std(repeat_scores) - base_score
            block_scores.append((block_name, importance_score_mean, importance_score_std))

        # Return output sorted by the mean ratio of change (second column)
        # For coalescing FPI output from many runs, we don't want this
        # sorting - just return the scores unsorted and we'll take the mean
        # over many models and then sort later.
        #return sorted(block_scores, key=lambda r: r[1], reverse=True)
        return block_scores
     
    def parse_permutation_feature_blocks(
        permutation_feature_blocks_str, df_column_index):
        
        blocks = [
            [bl_item.strip() for bl_item in bl.strip().split(',')]
            for bl in permutation_feature_blocks_str.strip().split(';')
        ]  if permutation_feature_blocks_str else list()

        column_idx = {v: k for k, v in enumerate(df_column_index)}
        blocks_ = list()
        blocks_names = list()
        explicit_blocks = set()
        for bl in blocks:
            parsed_block = list()
            for bl_item in bl:
                if ':' in bl_item:
                    start_col, end_col = bl_item.split(':')
                    parsed_block.extend(
                        list(df_column_index[column_idx[start_col]:column_idx[end_col] + 1]))
                else:
                    parsed_block.append(bl_item)
            blocks_.append(parsed_block)
            blocks_names.append(','.join(bl))
            explicit_blocks = explicit_blocks.union(set(parsed_block))

        for singleton in set(df_column_index) - explicit_blocks:
            blocks_.append([singleton])
            blocks_names.append(singleton)
        return blocks_, blocks_names
    
    #===========================================================
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

