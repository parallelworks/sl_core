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
    train_test_data = args.data
    predict_data = args.predict_data
    predict_var = args.predict_var
    
    # We know that the previous SuperLearner steps have
    # also generated the following files:
    predict_output = args.model_dir+"/sl_predictions.csv"
    
    # Finally, we output a final file that coalesces
    # all **available** sites for prediction and PCA
    # (any sites with NaN must be dropped from PCA)
    # That includes site ID, lon, lat, predicted
    # value, error metric, PCA dist, normalized
    # error, normalized PCA dist, and combined metric.
    pca_output = model_dir+"/sl_pca.csv"
    
    #======================================
    # Load files with Pandas and remove NaN
    #======================================
    # Load the features used to predict respiration rates:
    predict_inputs = pd.read_csv(predict_data)
    
    # Load the predicted respiration rates.  Store lon, lat, mean.error,
    # and predict.error for later. This information is only needed
    # for blending the error estimate with the PCA dist metric at the
    # very end.
    predict_targets = pd.read_csv(predict_output)
    
    # Check that the same number of sites are in both files
    print('Shapes at start in '+predict_inputs+' and '+predict_targets+':')
    print(predict_inputs.shape)
    print(predict_targets.shape)
    
    # Merge the two datasets now to cull missing data consistently across sites
    predict_all = pd.concat([predict_inputs,predict_targets],axis=1)
    
    # Check column transfer
    print('Shape after concat:')
    print(predict_all.shape)
    
    # Check for NaN:
    #print(np.sum(np.isnan(predict_all)))
    
    # Lots of missing values in oxygen, so drop all these.
    predict_all.drop(columns=['DO_mgL','DOSAT'],inplace=True)
    
    # 110 missing pH rows, drop whole rows.
    predict_all.dropna(axis=0,how='any',inplace=True)
    
    # Drop the targets, but keep them later for plotting.
    predict_rr = pd.DataFrame(
            predict_all.pop(predict_var),
            columns=pd.Index([predict_var]))
    
    # Remove lon, lat, and errors and store for later
    predict_xy = pd.DataFrame(predict_all.pop('lon'),columns=pd.Index(['lon']))
    predict_xy['lat'] = predict_all.pop('lat')
    
    predict_err = pd.DataFrame(predict_all.pop('mean.error'),columns=pd.Index(['mean.error']))
    predict_err['predict.error'] = predict_all.pop('predict.error')
    
    print('Shapes after NaN, x, y, error separation:')
    print(predict_all.shape)
    print(predict_xy.shape)
    print(predict_err.shape)
    print(predict_rr.shape)
    
    predict_xy.reset_index(drop=True,inplace=True)
    predict_err.reset_index(drop=True,inplace=True)
    predict_rr.reset_index(drop=True,inplace=True)
    predict_all.reset_index(drop=True,inplace=True)
    
    # Load the respiration rates used for training.  This should be in
    # exactly the same format as the merged predict_all but with fewer
    # rows.  Remove oxygen and respiration rates.
    training_all = pd.read_csv(train_test_data)
    training_all.drop(columns=['DO_mgL','DOSAT','rate.mg.per.L.per.h'],inplace=True)
    
    print('Training data shape:')
    print(training_all.shape)
    
    # Concatenate the training and prediction data sets for input to PCA.
    # Training data is at the head of the frame
    # Collab data is at the tail of the frame
    data_all = pd.concat([training_all,predict_all],axis=0)
    
    print('data_all shape after adding training data:')
    print(data_all.shape)
    data_all.reset_index(drop=True,inplace=True)
    
    # Remove any rows with any NaN in all columns execpt Gl_id
    search_nan_cols = []
    for col in data_all.columns:
        if col != 'GL_id':
            search_nan_cols.append(col)
            
    data_all.dropna(axis=0,how='any',inplace=True,subset=search_nan_cols)
    print('data_all shape after removing NaN:')
    print(data_all.shape)
    data_all.reset_index(drop=True,inplace=True)
    
    # Example for accessing the TRAINING DATA from the whole data set
    # (All training data points have NaN ID's.)
    #data_all[np.isnan(data_all['GL_id'])]
    
    # Example for accessing the COLLAB DATA from the whole data set
    # (All collab data have small IDs.)
    #data_all[data_all['GL_id'] < 10000]
    
    # Example for accessing the GLORICH DATA from the whole data set
    # (Prediction data have large IDs.)
    #data_all[data_all['GL_id'] > 10000]

    # We do not want the ID to be part of the PCA,
    # so pull it out now and concatenate it later as needed.
    id_df = pd.DataFrame(data_all.pop('GL_id'),columns=pd.Index(['GL_id']))
    
    #========================================
    # Scale data before PCA
    #========================================
    # cnsd = center, normalize by standard deviation
    cnsd = StandardScaler()
    cnsd.fit(data_all)
    data_all_scaled = cnsd.transform(data_all)
    
    # Can you recover the data from the scaler?  Yes!
    assert_array_almost_equal(data_all, cnsd.inverse_transform(data_all_scaled),decimal=6)
    
    # So overwrite the data to a scaled version
    data_all = data_all_scaled
    
    #=======================================
    # Run the PCA
    #=======================================
    # Data are automatically centered (no
    # need to subtract the mean).
    pca = PCA()
    pca.fit(data_all)

    # Find the loadings for each sample = how much of each component contributes to that sample.
    # [n_samples, n_features] dot TRANSPOSE([n_components, n_features]) = [n_samples, n_components]
    data_all_pca = pca.transform(data_all)

    # Plot the variance of each component to get a feel which components to keep
    fig, ax = plt.subplots()
    ax.plot(100*pca.explained_variance_ratio_,'b.-')
    ax.grid()
    print(np.sum(pca.explained_variance_ratio_))
    ax.set_xlabel('PCA component ID')
    ax.set_ylabel('Percent of variance explained')
    plt.savefig(model_dir+"/sl_pca_variance.png")

    #=======================================
    # 
    #=======================================
print("Done!")
