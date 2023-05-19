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

    # SuperLearner train.py always saves these files in model_dir:
    train_data = model_dir+'/train.csv'
    test_data = model_dir+'/test.csv'

    predict_data_csv = args.predict_data+'.csv'
    predict_data_ixy = args.predict_data+'.ixy'
    predict_output_file = model_dir+"/sl_predictions.csv"
    
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
    # Load the train and test data to make plot and estimate
    # prediction errors.
    #===========================================================
    num_inputs = int(args.num_inputs)

    train_df = pd.read_csv(train_data).astype(np.float32)
    X_train = train_df.values[:, :num_inputs]
    Y_train = train_df.values[:, num_inputs:]

    test_df = pd.read_csv(test_data).astype(np.float32)
    X_test = test_df.values[:, :num_inputs]
    Y_test = test_df.values[:, num_inputs:]
    
    all_df = pd.concat((train_df,test_df),axis=0)
    X_all = all_df.values[:, :num_inputs]
    Y_all = all_df.values[:, num_inputs:]
    
    #===========================================================
    # Make some predictions with the testing data
    #===========================================================
    Y_hat_train = superlearner[predict_var].predict(X_train)
    Y_hat_test = superlearner[predict_var].predict(X_test)

    # Compute line of best fit between testing and training targets
    test_line = np.polynomial.polynomial.Polynomial.fit(
        np.squeeze(Y_test),
        np.squeeze(Y_hat_test),1)
    test_xy = test_line.linspace(n=100,domain=[Y_all.min(),Y_all.max()])

    train_line = np.polynomial.polynomial.Polynomial.fit(
        np.squeeze(Y_train),
        np.squeeze(Y_hat_train),1)
    train_xy = train_line.linspace(n=100,domain=[Y_all.min(),Y_all.max()])

    # Compute some additional statistics
    # We want to see to what extent we can use
    # Y_hat_test to predict the error since we'll
    # be using Y_hat to estimate its own errors.
    # In this case, this is exactly what
    # the error in the mean and the predictions,
    # s_y and s_p, respectively, are getting at.
    # Equations based on McClave & Dietrich, 
    # _Statistics_, 6th Ed., pp. 672, 682, 707.
    n_sample_size = np.size(Y_test)
    x_bar = np.mean(Y_test)
    y_bar = np.mean(Y_hat_test)
    ssxx = np.sum((np.squeeze(Y_test)-np.mean(Y_test))**2)
    ssyy = np.sum((np.squeeze(Y_hat_test)-np.mean(Y_hat_test))**2)
    ssxy = np.sum((np.squeeze(Y_hat_test)-np.mean(Y_hat_test))*(np.squeeze(Y_test)-np.mean(Y_test)))
    
    # Root squared errors
    rse = np.sqrt((np.squeeze(Y_test)-Y_hat_test)**2)

    # When trying to predict error based on Y_hat,
    # the sse_error = sum((error_i - mean(error))^2)
    # which when expanded algebreically
    sse_error = ssxx + ssyy - 2.0*ssxy
    
    # Estimator for the standard error of regression between
    # true targets and predicted targets.
    sse = sse_error # Do this if using the test predictions to estimate the error
    s = np.sqrt(sse/(n_sample_size-2))
    
    # Estimate of the sampling distribution of the predictions
    # of the mean value of the targets at specific value.
    # This is nice, but it's very flat -> does not change much
    # based on Y_test.
    ssxx = ssyy # (Include this line too for Y_hat_test as a predictor of error, otherwise Y_test is the predictor.)
    s_y = 2*s*np.sqrt((1/n_sample_size) + ((np.squeeze(Y_test)-np.mean(Y_test))**2)/ssxx)
    s_p = 2*s*np.sqrt(1+(1/n_sample_size) + ((np.squeeze(Y_test)-np.mean(Y_test))**2)/ssxx)

    fig, ax = plt.subplots()
    ax.plot(Y_test,rse,'ko')
    ax.plot(Y_test,s_y,'ro')
    ax.plot(Y_test,s_p,'co')
    ax.plot(Y_test,s*np.ones(np.shape(Y_test)),'r.')
    ax.set_ylabel('Error metric [mg O2/L/h]')
    ax.set_xlabel('Target respiration rate [mg O2/L/h]')
    ax.legend(['RSE','Error in mean','Prediction Error','Regression Error'],loc='lower left')
    ax.grid()
    plt.savefig(model_dir+'/sl_error.png')
    
    # Print out correlations between the various error estimates:
    print("RSE: "+str(np.mean(rse))+" +/- "+str(np.std(rse)))
    print("mean.error: "+str(np.mean(s_y))+" +/- "+str(np.std(s_y)))
    print("predict.error: "+str(np.mean(s_p))+" +/- "+str(np.std(s_p)))
    
    print("RSE vs mean.error: "+str(np.corrcoef(rse,s_y)))
    print("RSE vs predict.error: "+str(np.corrcoef(rse,s_p)))
    # Causes workflow to crash, not very useful, comment out.
    #print("RSE vs s: "+str(np.corrcoef(rse,s*np.ones(np.shape(Y_test)))))
    
    #===========================================================
    # Make an evaluation plot
    #===========================================================
    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(Y_train,np.squeeze(Y_hat_train),'ko',markersize=10)
    ax.plot(Y_test,np.squeeze(Y_hat_test),'ko',markersize=10,fillstyle='none')
    ax.plot(test_xy[0],test_xy[1],'r-')
    ax.plot(train_xy[0],train_xy[1],'r--')
    ax.grid()
    ax.set_xlabel('Original target values, mg O2/L/h')
    ax.set_ylabel('Model predictions, mg O2/L/h')
    
    # Predict and metric with the individual models
    # (The same thing can be achieved with built in:
    # superlearner[predict_var].transform(X)
    # but done explicitly here.)
    for model_name in list_models:
        model_object = superlearner[predict_var].named_estimators_[model_name]
        Y_hat_train_mod = model_object.predict(X_train)
        Y_hat_test_mod = model_object.predict(X_test)
        
        # Color coded dots
        ax.plot(np.concatenate((Y_train,Y_test),axis=0),
                np.concatenate((np.squeeze(Y_hat_train_mod),np.squeeze(Y_hat_test_mod)),axis=0),
                '.',markersize=5)
    
    # One-to-one line
    ax.plot(Y_all,Y_all,'k')

    # Set zoom
    ax.set_xlim([-45,0])
    ax.set_ylim([-45,0])

    # Legend
    ax.legend(['Stacked TRAIN','Stacked TEST','TEST corr','TRAIN corr']+list_models+['one-to-one'])

    # Save figure to file
    plt.savefig(model_dir+'/sl_scatter.png')

    #===========================================================
    # Make predictions with a large data set
    #===========================================================
    predict_df = pd.read_csv(predict_data_csv).astype(np.float32)
    predict_df.fillna(predict_df.mean(),inplace=True)
    X = predict_df.values
    
    Y_predict = superlearner[predict_var].predict(X)
    
    # Estimate the error based on the training data
    Y_hat_error = 2*s*np.sqrt((1/n_sample_size) + ((np.squeeze(Y_predict)-np.mean(Y_test))**2)/ssxx)
    Y_hat_pred_error = 2*s*np.sqrt(1+(1/n_sample_size) + ((np.squeeze(Y_predict)-np.mean(Y_test))**2)/ssxx)
    
    #===========================================================
    # Write output file
    #===========================================================
    
    # Put the predictions with lon lat data separated beforehand.
    output_df = pd.read_csv(predict_data_ixy)
        
    output_df[predict_var] = pd.Series(Y_predict)
    output_df['mean.error'] = pd.Series(Y_hat_error)
    output_df['predict.error'] = pd.Series(Y_hat_pred_error)
    output_df.to_csv(
        predict_output_file,
        index=False,
        na_rep='NaN')

print("Done!")
