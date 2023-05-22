#=====================================
# SuperLearner core execution script
#=====================================
# Fit Scikit-Learn StackingRegressor
# sub-models in parallel.  Also run
# hyperparameter optimization on
# sub-models in parallel.
#
# Command line execution expects the
# following arguments:
# sl_main.py
# --conda_sh '/tmp/pworks/.miniconda3/etc/profile.d/conda.sh'
# --superlearner_conf '/pw/workflows/sl_test/superlearner_conf.py'
# --n_jobs '8'
# --num_inputs '25'
# --cross_val_score 'True'
# --model_dir './model_dir'
# --hpo 'True'
# --smogn 'True'
# --data '/pw/workflows/sl_test/whondrml_global_train_25_inputs_update.csv'
# --backend 'loky'
#
# Caveats:
# If the training data is too big, fitting
# may fail due to memory issues that are
# NOT caught in the logs!
#=====================================
# Load dependencies
#=====================================
import sklearn
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
#from imblearn.over_sampling import RandomOverSampler
#from imblearn.under_sampling import RandomUnderSampler

import joblib; print(joblib.__version__)

import pandas as pd
import math
import numpy as np
import smogn

import importlib

import sys
import argparse
import os, shutil, pickle, json
from copy import deepcopy
import random

# For data plots
import matplotlib.pyplot as plt

#=======================================
# Supporting functions
#=======================================

def clean_data_df(data_df):
    # Outlier checking could be done here but
    # is not implemented.

    # Fill any NAN with the mean
    print("NOTICE: Filling any NaN with mean values!")
    print("NaN in training data will cause a crash.")
    data_df.fillna(data_df.mean(), inplace = True)
    return data_df

def load_data_csv_io(data_csv, num_inputs):
    # Read from input CSV file (data_csv is a string
    # with the path of the file) assuming that all the
    # features (i.e. inputs) to the model are in the
    # leftmost columns while the targets (i.e. outputs
    # of the model are in the rightmost colums.  The
    # feature/target split is defined by the integer
    # num_inputs.

    # Load whole data set
    data_df = pd.read_csv(data_csv).astype(np.float32)

    # Fill NaN and possibly remove outliers
    data_df = clean_data_df(data_df)

    # Get names of the data columns (features and targets)
    pnames = list(data_df)

    # Split features (X) and targets (Y)
    X = data_df.values[:, :num_inputs]
    Y = data_df.values[:, num_inputs:]
    
    return X, Y, pnames[:num_inputs], pnames[num_inputs:]

def load_data_df_io(data_df, num_inputs):
    # Split features and targets from a dataframe based on
    # number of inputs.
    data_df = clean_data_df(data_df)
    pnames = list(data_df)
    X = data_df.values[:, :num_inputs]
    Y = data_df.values[:, num_inputs:]
    return X, Y, pnames[:num_inputs], pnames[num_inputs:]

def save_data_csv_io(x_np, y_np, inames, onames, out_file_name):
    # Concatenate X (features) and Y (targets) into single CSV
    x_df = pd.DataFrame(x_np,columns=inames)
    y_df = pd.DataFrame(y_np,columns=onames)
    df_out = pd.concat([x_df,y_df],axis=1)
    df_out.to_csv(out_file_name,index=False,na_rep='NaN',mode='w')

def format_estimators(estimators_dict):
    # Define StackingRegressor
    estimators = []
    for est_id, est_conf in estimators_dict.items():
        estimators.append((est_id, est_conf['model']))
    return estimators

#=======================================
# Main execution
#=======================================
if __name__ == '__main__':

    #===========================
    # Command line inputs
    #===========================
    print("Parsing SuperLearner input arguments...")
    parser = argparse.ArgumentParser()
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg)
            print(arg)

    args = parser.parse_args()
    
    if args.backend == 'dask':
        n_jobs = int(args.n_jobs)
        # FIXME: Make this code common
        from dask_jobqueue import SLURMCluster
        import dask
        from dask.distributed import Client

        # Log dir needs to be accessible to the compute nodes too!!!!
        dask_log_dir = '/contrib/dask-logs/'
        cluster = SLURMCluster(
            cores = int(args.cores),
            memory= str(args.memory),
            walltime= '00:55:00',
            log_directory= dask_log_dir,
            env_extra= ['source ' + args.conda_sh + '; conda activate']
        )
        cluster.adapt(minimum = 0, maximum = n_jobs)
        client = Client(cluster)
        backend_params = {'wait_for_workers_timeout': 600}
    else:
        backend_params = {}
        n_jobs = None

    predict_var=args.predict_var

    #===========================
    # Create Model Directory
    #===========================
    args.model_dir = args.model_dir.replace('*','')
    os.makedirs(args.model_dir, exist_ok = True)

    #===========================
    # Load Data
    #===========================

    # Set same seed (test upper bound, below)
    #SEED = 1000000

    # Set random seed
    SEED = random.randint(1,1000000)

    #data = pd.read_csv(args.data).astype(np.float32)
    #data = clean_data_df(data)
    # Shuffle the entire dataset
    #data = data.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Always load the original data for cross-validation
    X, Y, inames, onames = load_data_csv_io(args.data, int(args.num_inputs))
    save_data_csv_io(X, Y, inames, onames, args.model_dir+'/original_input_data.csv')

    # Train and test dataset construction depends on SMOGN or not
    if args.smogn == "True":
    
        data = pd.read_csv(args.data).astype(np.float32)
        data = clean_data_df(data)
        # Shuffle the entire dataset
        data = data.sample(frac=1, random_state=SEED).reset_index(drop=True)

        # Remove 25% of the dataset for testing later
        # 25% is the default in sklearn.train_test_split
        # This is essential since information from data points 
        # input to SMOGN will transfer throughout the resulting
        # training set. 
        smogn_test = data.iloc[:math.ceil(len(data)*.25), :]
        smogn_train = data.iloc[math.ceil(len(data)*.25):, :].reset_index()

        # Remove this so we only have one test data set.
        #smogn_test.to_csv(args.model_dir+"/smogn_test.csv", index=False, na_rep='NaN')

        # Apply SMOGN

        # specify phi relevance values
        # rg_mtrx = [
        #     [0,  1, 0],  ## over-sample ("minority")
        #     [-10, 0, 0],  ## under-sample ("majority")
        #     [-20, 0, 0],  ## under-sample
        #     [-30, 0, 0],  ## under-sample
        # ]

        regular_smogn = smogn_train
        extreme_smogn = smogn_train
        y_col_name = predict_var

        # number of smogn iterations 
        # TODO: high iterations throws an error with either duplicate values or not enough datapoints 
        for i in range(1):
            regular_smogn = regular_smogn.append(smogn.smoter(
                data = regular_smogn,
                y = y_col_name,
                drop_na_row = True,
                seed=SEED
            ))
            regular_smogn = regular_smogn.drop_duplicates()

            extreme_smogn = extreme_smogn.append(smogn.smoter(
                data = extreme_smogn,
                y = y_col_name,
                drop_na_row = True,
                # rel_method = 'manual',    ## string ('auto' or 'manual')
                # rel_ctrl_pts_rg = rg_mtrx, ## 2d array (format: [x, y])
                samp_method = "extreme",
                seed=SEED
            ))
            extreme_smogn = extreme_smogn.drop_duplicates()

        final_smogn_train = regular_smogn.append(extreme_smogn)
        final_smogn_train = final_smogn_train.drop_duplicates()
        final_smogn_train = final_smogn_train.drop(columns=["index"])
        #final_smogn_train.to_csv(args.model_dir + "/final_smogn_train.csv", index=False, na_rep='NaN')

        # process data for the superlearner
        X_test, Y_test, inames_test, onames_test = load_data_df_io(smogn_test,int(args.num_inputs))
        X_train, Y_train, inames_train, onames_train = load_data_df_io(final_smogn_train, int(args.num_inputs))
    else:
        # Indent so it is clear that train_test_split is only used for non-SMOGNed data
        # NOTE: train and test datasets are the same size
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=SEED)

    # Apply sampling to training set only
    # CAUTION: Template for testing only.
    # Currently works only if Y is an integer
    # which is interpreted as a class number
    # by imblearn.  In the future, generalize this
    # by including a non-target column that is
    # the class of each row so imblearn can be
    # applied more generally.
    #ros = RandomOverSampler()
    #rus = RandomUnderSampler()
    #X_train, Y_train = rus.fit_resample(X_train, Y_train)
    #Y_train = np.expand_dims(Y_train, axis=1)

    #===========================
    # Load SuperLearner config as a package
    #===========================
    print("Loading SuperLearner configuration...")
    print("Loading from: "+args.superlearner_conf)
    print("Dirname:  "+os.path.dirname(args.superlearner_conf))
    print("Basename: "+os.path.basename(args.superlearner_conf.replace('.py','')))

    # Add config's dir to the path
    sys.path.append(os.path.dirname(args.superlearner_conf))
    sl_conf = getattr(
        # Second, import the file as a module.  Drop ".py".
        importlib.import_module(os.path.basename(args.superlearner_conf.replace('.py',''))),
        'SuperLearnerConf'
    )

    # The SL configuration file is needed to load the SL pickle
    try:
        # To prevent same file error!
        shutil.copy(args.superlearner_conf, args.model_dir)
    except:
        pass # FIXME: Add error handling!

    #================================
    # Run hyperparameter optimization
    #================================
    if args.hpo == "True":
        sl_conf_hpo = deepcopy(sl_conf)
        sl_conf_hpo['estimators'] = {}
        for oi, oname in enumerate(onames):
            sl_conf_hpo['estimators'][oname] = {}

            if oname in sl_conf['estimators']:
                estimators = sl_conf['estimators'][oname]
            else:
                estimators = sl_conf['estimators']

            for ename, einfo in estimators.items():
                print('Running HPO for output {} and estimator {}'.format(oname, ename), flush = True)
                sl_conf_hpo['estimators'][oname][ename] = {}

                if 'hpo' not in einfo:
                    sl_conf_hpo['estimators'][oname][ename]['model'] = einfo['model']

                with joblib.parallel_backend(args.backend, **backend_params):
                    sl_conf_hpo['estimators'][oname][ename]['model'] = einfo['hpo'].fit(X_train, Y_train[:, oi]).best_estimator_

        sl_conf = sl_conf_hpo

    #========================
    # Define SuperLearners
    #========================
    SuperLearners = {}
    for oi, oname in enumerate(onames):
        print('Defining estimator for output: ' + oname, flush = True)

        if oname in sl_conf['estimators']:
            estimators = sl_conf['estimators'][oname]
        else:
            estimators = sl_conf['estimators']

        final_estimator = sl_conf['final_estimator']
        if type(final_estimator) == dict:
            if 'oname' in sl_conf['final_estimator']:
                final_estimator = sl_conf['final_estimator'][oname]


        SuperLearners[oname] = StackingRegressor(
            estimators = format_estimators(estimators),
            final_estimator = final_estimator,
            n_jobs = n_jobs
        )

    #=================================================================
    # Fit SuperLearners:
    for oi, oname in enumerate(onames):
        print('Training estimator for output: ' + oname, flush = True)
        with joblib.parallel_backend(args.backend, **backend_params):
            SuperLearners[oname] = SuperLearners[oname].fit(X_train, Y_train[:, oi])


    with open(args.model_dir + '/SuperLearners.pkl', 'wb') as output:
        pickle.dump(SuperLearners, output, pickle.HIGHEST_PROTOCOL)
    
    #================================================================
    # Cross_val_score:
    if args.cross_val_score == "True":
        cross_val_metrics = {}
        for oi, oname in enumerate(onames):
            cross_val_metrics[oname] = dict.fromkeys(['all', 'mean', 'std'])
            # FIXME: dask bug with cross_val_score!
            with joblib.parallel_backend('threading', **{}):
                scores = cross_val_score(
                    deepcopy(SuperLearners[oname]),
                    X,
                    y = Y[:, oi],
                    n_jobs = n_jobs
                )

            cross_val_metrics[oname]['all'] = list(scores)
            cross_val_metrics[oname]['mean'] = scores.mean()
            cross_val_metrics[oname]['std'] = scores.std()

        print('Cross-validation metrics:', flush = True)
        print(json.dumps(cross_val_metrics, indent = 4), flush = True)
        with open(args.model_dir + '/cross-val-metrics.json', 'w') as json_file:
            json.dump(cross_val_metrics, json_file, indent = 4)
        print('Statistics of the cross-validation metrics:')


    #===========================================================
    # Evaluate SuperLearners on test set:
    ho_metrics = {}
    for oi, oname in enumerate(onames):
        print('Evaluating estimator for output: ' + oname, flush = True)
        with joblib.parallel_backend(args.backend, **backend_params):
            ho_metrics[oname] = SuperLearners[oname].score(X_test, Y_test[:, oi])

    print('Hold out metrics:', flush = True)
    print(json.dumps(ho_metrics, indent = 4), flush = True)
    with open(args.model_dir + '/hold-out-metrics.json', 'w') as json_file:
        json.dump(ho_metrics, json_file, indent = 4)

    #============================================================
    # Evaluate SuperLearners on training set ("classical validation"):
    ho_metrics = {}
    for oi, oname in enumerate(onames):
        print('Evaluating estimator for output: ' + oname, flush = True)
        with joblib.parallel_backend(args.backend, **backend_params):
            ho_metrics[oname] = SuperLearners[oname].score(X_train, Y_train[:, oi])

    print('Classical metrics:', flush = True)
    print(json.dumps(ho_metrics, indent = 4), flush = True)
    with open(args.model_dir + '/classical-metrics.json', 'w') as json_file:
        json.dump(ho_metrics, json_file, indent = 4)

    #=========================================================
    # For debugging
    if args.backend == 'dask':
        shutil.move(dask_log_dir, args.model_dir)

    #============================================================
    # Save the training and testing data seprately for evaluation
    # (They are randomly split above.)
    print('Save train/test data')
    save_data_csv_io(X_train, Y_train, inames, onames, args.model_dir+'/train.csv')
    save_data_csv_io(X_test, Y_test, inames, onames, args.model_dir+'/test.csv')

    #============================================================
    # Make a plot showing the histograms of the training and testing
    # data sets.
    fig, ax = plt.subplots(figsize=(15,6))
    n, bins, patches = ax.hist(Y_train, 20, density=False, facecolor='g', alpha=0.5, align='mid', histtype='stepfilled')
    n, bins, patches = ax.hist(Y_test, 20, density=False, facecolor='k', alpha=0.5, align='mid', histtype='stepfilled')
    ax.legend(['Training set','Testing set'])
    ax.set_xlabel('Histogram of target')
    ax.set_ylabel('Frequency')
    plt.savefig(args.model_dir+'/sl_targets_hist.png')

