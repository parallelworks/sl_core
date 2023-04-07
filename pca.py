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
from numpy.testing import assert_array_almost_equal
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
    predict_inputs = pd.read_csv(predict_data+".csv")
    
    # Load the predicted respiration rates.  Store lon, lat, mean.error,
    # and predict.error for later. This information is only needed
    # for blending the error estimate with the PCA dist metric at the
    # very end.
    predict_targets = pd.read_csv(predict_output)
    
    # Check that the same number of sites are in both files
    print('Shapes at start:')
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
    predict_all.drop(
        columns=predict_all.columns[
            predict_all.columns.str.contains('DO')],
            inplace=True)
    
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
    training_all.drop(
        columns=training_all.columns[
            np.logical_or(
                training_all.columns.str.contains('DO'),
                training_all.columns.str.contains(predict_var))],
        inplace=True)
    
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

    # Finally, the training data did not have an ID but it does
    # have quite a few NaN in it.  Overwrite those NaN with mean
    # values (as done for the training) and save the dataframe
    # so we can estimate the PCA-distance for each datapoint in
    # the training set, separately.
    training_all.fillna(training_all.mean(), inplace = True)
    
    #========================================
    # Scale data before PCA
    #========================================
    # cnsd = center, normalize by standard deviation
    cnsd = StandardScaler()
    cnsd.fit(data_all)
    data_all_scaled = cnsd.transform(data_all)
    training_all_scaled = cnsd.transform(training_all)
    
    # Can you recover the data from the scaler?  Yes!
    assert_array_almost_equal(data_all, cnsd.inverse_transform(data_all_scaled),decimal=6)
    
    # So overwrite the data to a scaled version
    data_all = data_all_scaled
    training_all = training_all_scaled
    
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
    training_all_pca = pca.transform(training_all)
    
    # Plot the variance of each component to get a feel which components to keep
    fig, ax = plt.subplots()
    ax.plot(100*pca.explained_variance_ratio_,'b.-')
    ax.grid()
    print(np.sum(pca.explained_variance_ratio_))
    ax.set_xlabel('PCA component ID')
    ax.set_ylabel('Percent of variance explained')
    plt.savefig(model_dir+"/sl_pca_variance.png")
    
    #=======================================
    # Find the PCA distance for each point
    #=======================================
    # Merge the GL_id back into the PCA'ed data for indexing
    # All training data (WHONDRS) - GL_id = NaN
    # All collab data - GL_id < 100000
    data_all_pca_w_id = pd.concat([id_df,pd.DataFrame(data_all_pca)],axis=1)
    
    # Using just the first two components:
    # (Not significantly different from using all components)
    pca_n2_dist = np.linalg.norm(data_all_pca[:,0:2],axis=1)
    
    # Using all components (unweighted)
    pca_all_dist = np.linalg.norm(data_all_pca,axis=1)
    
    # 1) Get the WHONDRS PCA data and get collab PCA data
    # (square brackets at the end trim off the ID column while
    # retaining the PCA component data).
    data_WHONDRS_pca = data_all_pca_w_id[np.isnan(data_all_pca_w_id['GL_id'])].values[:,1:]
    data_collab_pca = data_all_pca_w_id[data_all_pca_w_id['GL_id'] < 10000].values[:,1:]
    
    # 2) Get the WHONDRS centroid:
    WHONDRS_centroid = data_WHONDRS_pca.mean(0)
    
    # 3) Get the distance wrt WHONDRS centroid using
    #    only first two PCA components:
    pca_n2_WHONDRS_dist = np.linalg.norm(
        (data_all_pca[:,0:2] - WHONDRS_centroid[0:2]),axis=1)
    training_n2_WHONDRS_dist = np.linalg.norm(
        (training_all_pca[:,0:2] - WHONDRS_centroid[0:2]),axis=1)
    
    # 4) Plots
    # Compare all components to 2 components
    fig, (ax0, ax1, ax2) = plt.subplots(1,3,figsize=(15,5))
    ax0.plot(pca_n2_dist,pca_all_dist,'k.')
    ax0.set_xlabel('Site distance using 2 components')
    ax0.set_ylabel('Site distance using all components')
    ax0.plot([0,15],[0,15],'r-')
    ax0.grid()
    
    # Compare all-data centroid to WHONDRS centroid
    ax1.plot(pca_n2_dist,pca_n2_WHONDRS_dist,'k.')
    ax1.set_xlabel('Site distance using 2 components and all sites` centroid')
    ax1.set_ylabel('Site distance using 2 components and WHONDRS centroid')
    ax1.plot([0,15],[0,15],'r-')
    ax1.grid()
    
    # Plot PC1 v.s. PC2
    ax2.plot(data_all_pca[:,0],data_all_pca[:,1],'r+')
    ax2.plot(data_WHONDRS_pca[:,0],data_WHONDRS_pca[:,1],'k.')
    ax2.plot(data_collab_pca[:,0],data_collab_pca[:,1],'c+',markerfacecolor="none", markersize="10", markeredgecolor="cyan")
    #ax2.plot(training_all_pca[:,0],training_all_pca[:,1],'ko')
    
    # Do NOT need to plot all data centroid - it is zero by definition
    ax2.plot(WHONDRS_centroid[0],WHONDRS_centroid[1],'k+',markerfacecolor="none", markersize="30", markeredgecolor="black")
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.grid()
    ax2.legend(['All data points','WHONDRS sites','Collab sites','WHONDRS centroid'])
    plt.savefig(model_dir+"/sl_pca_scatter.png")
    
    #================================
    # Final merge and write output
    #================================
    # Get just the IDs for the predict points (GLORICH + COLLAB)
    id_predict_df = id_df[np.logical_not(np.isnan(id_df['GL_id']))]
    
    # Concatenate the PCA distance with the errors
    predict_err['pca.dist'] = pd.DataFrame(
        pca_n2_WHONDRS_dist[
            np.logical_not(
                np.isnan(data_all_pca_w_id['GL_id']))])
    
    # Append the PCA dist to the training data
    # (We don't need this every time - use it to
    # decide on order to add WHONDRS sites for the
    # experimental series of ModEx iterations.)
    # (Actual file write out is commented out,
    # below.)
    training_all = pd.read_csv(train_test_data)
    training_all['pca.dist'] = pd.DataFrame(
        training_n2_WHONDRS_dist)
    
    # Normalize and combine PCA distance with error
    predict_err['mean.error.scaled'] = predict_err['mean.error']/predict_err.max()['mean.error']
    predict_err['pca.dist.scaled'] = predict_err['pca.dist']/predict_err.max()['pca.dist']
    predict_err['combined.metric'] = predict_err['mean.error.scaled']*predict_err['pca.dist.scaled']

    # Ensure all dataframe indeces are restarted
    id_predict_df.reset_index(drop=True,inplace=True)
    predict_err.reset_index(drop=True,inplace=True)
    predict_rr.reset_index(drop=True,inplace=True)
    predict_xy.reset_index(drop=True,inplace=True)

    # Put all output into a single dataframe
    output_df = pd.concat([
        id_predict_df,
        predict_xy,
        predict_rr,
        predict_err],axis=1)

    # Write to output files
    output_df.to_csv(pca_output,index=False)
    #training_all.to_csv(pca_output.split(".")[0]+"_training.csv",index=False)
print("Done!")
