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
import seaborn as sns

import sys

#=======================================
# Main execution
#=======================================
if __name__ == '__main__':

    # Constants
    # TODO: we don't use jobid in local_superlearner_test.sh
    jobid = 0

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

    # Finally, the list of weights of the models can be accessed with:
    # (Note the _ are important and have to do with scikit learn
    # naming conventions of variables that are set after fitting.)
    sl_weights = superlearner[predict_var].final_estimator_.weights_

    # Use the weights to get a list of the models 
    # that have been included in the SuperLearner
    sl_models = []
    mm = 0
    for model in list_models:
        #print(model)
        sl_weights[mm]
        if ( sl_weights[mm] > 0.1 ):
            sl_models.append(model)
        mm = mm + 1
    
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

    all_df = pd.concat((train_df,test_df),axis=0)

    # Pull the target column out, and remove it from data_df.
    target_train_df = train_df.pop(predict_var)
    target_test_df = test_df.pop(predict_var)
    target_all_df = all_df.pop(predict_var)

    # Lines below may need to be generalized for multi-var
    # predictions.
    X_predict = pd.read_csv(predict_data_csv).astype(np.float32)
    tmp_df = pd.read_csv(predict_output_file, dtype={'Sample_ID': str})
    Y_predict = tmp_df[predict_var]
    
    #==========================================================
    # FPI functions
    #==========================================================

    #----------------------------------------------------------
    def permute_importance(permutation_feature_blocks_str, model, X, y, scoring_func, n_repeats=20, ratio_score=True):

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
    
    #----------------------------------------------------------
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
    
    #----------------------------------------------------------
    # FPI only works if correlated features are permuted together.
    # Otherwise, correlated features permuted independently
    # will dilute the impact of that feature since the ML model
    # will still get some information from the unpermuted feature.
    #
    # --- Conventions ---
    # Within each group of features, the feature names are separated by commas.
    # The groups of features are separated by semi-colons.
    # Colons can be used for contiguous feature grouping, but this currently
    # ignored because need to come up with a reliable way to generalize processing
    # this case since it assumes the same feature names throughout.
    #
    # Inputs: Takes a list of feature names and a correlation heatmap between features
    # (One could just pass a .csv file and compute the correlation internally, but keep
    # separate for now to enable plotting and debugging.)
    def group_correlated_features(
        feature_corr,
        corr_cutoff=0.4,
        merge_groups=False,
        onehot_list=[],
        verbose=False):

        # Take absolute value of the correlations.
        abs_corr = np.abs(feature_corr)

        # First get rid of diagonal (but allowing for 1.0 correlations 
        # elsewhere, e.g. duplicate features)
        feature_names_str = feature_corr.columns
        for name in feature_names_str:
            abs_corr.loc[name,name] = 0

        # Initialize tracker for highest correlation detected
        current_highest_corr = 1.0

        # Initialize list of groups
        # (Simple approach, ignoring one-hot features)
        feature_groups_list = []

        # For one-hot features, we want to ensure all one-of-k streams
        # are all permuted together.  This means that we pre-populate
        # the feature_groups_list with the user specified one-hot 
        # features.  These features are identified by their prefix.
        # If a prefix is in the list, a new group is created and
        # all features in the list that match the prefix
        # are automatically included in the group.  Later, if one
        # (or more) of the one-hot streams in each one-hot feature
        # correlates with another feature, those features can be
        # merged, but all the streams from a one-hot feature will
        # be carried as a block.
        for prefix in onehot_list:

            if verbose:
                print('Finding one-hot features based on given prefix: '+prefix)
            # Create a new group for elements found with this prefix.
            prefix_match = [s for s in feature_names_str if prefix in s]

            jj = 0
            for feature in prefix_match:
                if jj == 0:
                    if verbose:
                        print('Creating one-hot group for feature: '+feature)
                    # Create the group with first feature
                    feature_groups_list.append(feature)
                    jj = 1
                else:
                    if verbose:
                        print('Appending to one-hot group: '+feature)
                    feature_groups_list[-1] = feature_groups_list[-1]+','+feature

        # Initialize a counter to track number of times doing this
        ii = 1

        while current_highest_corr >= corr_cutoff :

            # Scalar highest correlation in DF, could be duplicated
            current_highest_corr = abs_corr.max().max()

            # Get locations of current_highest_corr, DF True at current_highest_corr
            bool_current_highest_corr = abs_corr==current_highest_corr

            # Get DF of NaN except 1's at locations of current_highest_corr
            loc_current_highest_corr = abs_corr[bool_current_highest_corr]/current_highest_corr

            # Count number of highest correlations in DF.  Divide by
            # two because the corr DF is symmetric.
            num_highest_corr = np.nansum(loc_current_highest_corr)/2

            if verbose:
                print('Found highest correlation: '+str(current_highest_corr)+' with '+str(num_highest_corr)+' instances, iteration '+str(ii))

            # Find the features involved in this correlation
            features_in_corr = []
            index_current_highest_corr = np.where(bool_current_highest_corr)
            for jj in index_current_highest_corr :
                for kk in jj :
                    features_in_corr.append(abs_corr.index[kk])

            # Get a unique list of features in correlation
            features_in_corr = list(set(features_in_corr))
            features_in_corr_group_id = np.full(np.shape(features_in_corr),np.nan)

            # Check if any features_in_corr are already assigned in a group
            for fid,feature in enumerate(features_in_corr) :
                # Search for this feature in feature_groups_list
                for gid,group in enumerate(feature_groups_list):
                    for feature_in_group in group.split(','):
                        if (feature == feature_in_group):
                            features_in_corr_group_id[fid] = gid

            if verbose:
                print('Found correlation:-----------------------------------------------')
                print(features_in_corr)
                print('Starting gids:---------------------------------------------------')
                print(features_in_corr_group_id)
                print('Starting feature list:-------------------------------------------')
                print(feature_groups_list)

            if np.nansum(np.isfinite(features_in_corr_group_id)) == 0:
                if verbose:
                    print('No features already in group.')
                # None of the features in this corr are in a grouping so
                # make a new group with all features.
                for fid,feature in enumerate(features_in_corr):
                    if fid == 0:
                        # Create group for first feature
                        if verbose:
                            print('Creating new group----------------------------<===')
                        feature_groups_list.append(feature)
                    else:
                        # Add features to the group
                        feature_groups_list[-1] = feature_groups_list[-1]+','+feature

            elif np.nansum(np.isfinite(features_in_corr_group_id)) == 1:
                if verbose:
                    print('Exactly one feature already in group.')
                # Exactly one of the features is already in a group so
                # Add the other features (with group id == NaN) to that group.
                # Find the group id of the ONE feature already in a group:
                gid = int(np.nansum(features_in_corr_group_id))
                for fid,feature in enumerate(features_in_corr):
                    if np.isnan(features_in_corr_group_id[fid]):
                        feature_groups_list[gid] = feature_groups_list[gid]+','+feature
            else:
                # There is some combination of at least two 
                # features in existing groups and some other
                # feaures may or may not be in groups.  Some
                # of the features could already be in the same
                # group.  For cases when there there always exactly
                # one correlation pair detected, then no new groups
                # are created with merge_groups=False because the
                # two correlated features already belong to different
                # groups.
                if np.all(features_in_corr_group_id == features_in_corr_group_id[0]):
                    # Special case when all features detected are 
                    # already assigned to the same group.
                    if verbose:
                        print('All features already in same group. Do nothing.')
                elif merge_groups:
                    if verbose:
                        print('Merge all groups/new features for this correlation.')
                    # Merge any groups associated with any feature detected here.
                    jj = 0
                    gid_merged = []
                    for fid,feature in enumerate(features_in_corr):
                        gid = features_in_corr_group_id[fid]
                        if verbose:
                            print('For feature: '+feature+' with gid: '+str(gid))
                        if (jj == 0):
                            # Always create a new group for the supergroup we're
                            # about to build.
                            if verbose:
                                print('Creating new group----------------------------<===')
                            if np.isnan(gid):
                                # The first feature does not have a group, so create
                                # the new supergroup with just it.
                                if verbose:
                                    print('New group is feature: '+feature)
                                feature_groups_list.append(feature)
                            else:
                                # Convert gid to int now that we know that gid is not NaN
                                gid = int(gid)

                                # The first feature has a group, so create the new supergroup
                                # as a duplicate of the first feature's group.
                                if verbose:
                                    print('New group is duplicate group: '+feature_groups_list[gid])
                                feature_groups_list.append(feature_groups_list[gid])

                                # Keep track of which groups have already been merged.
                                gid_merged.append(gid)
                            jj = 1
                        else:
                            # The new supergroup exists, so append group/feature names
                            # if it hasn't been appended before.
                            if np.isnan(gid):
                                # Feature with no group, append to supergroup
                                if verbose:
                                    print('Feature with no group -> simple append')
                                    print('Appending feature: '+feature+' to group: '+feature_groups_list[-1])
                                feature_groups_list[-1] = feature_groups_list[-1]+','+feature
                            else:
                                # Convert gid to int now that we know that gid is not NaN
                                gid = int(gid)

                                if gid in gid_merged:
                                    # Feature with a group but group has already been merged.
                                    if verbose:
                                        print('Group already appended, skip this feature.')
                                else:
                                    # Feature with a group, group has not already 
                                    # been appended.  Append now.
                                    if verbose:
                                        print('Group not already appended.')
                                        print('Appending group: '+feature_groups_list[gid]+' to group: '+feature_groups_list[-1])
                                    feature_groups_list[-1] = feature_groups_list[-1]+','+feature_groups_list[gid]
                                    gid_merged.append(gid)
                    # Now that we are done creating the new supergroup, clean up
                    # by deleting the existing groups that have been merged into
                    # the supergroup.  Note that removal of merged groups MUST
                    # proceed in sorted, high GID to low GID because the length
                    # of the list is changed with pop operations, so the GIDs are
                    # reset.  Also add list(set()) inside to ensure that if there
                    # are duplicate groups being merged, there is only one group
                    # delete operation.
                    if verbose:
                        print('Delete merged groups.')
                        print(gid_merged)
                        print(list(set(gid_merged)))
                    for gid in sorted(list(set(gid_merged)), reverse=True):
                        if verbose:
                            print('Deleting merged group '+feature_groups_list[gid]+'----------------------<===')
                        feature_groups_list.pop(gid)

                else:
                    if verbose:
                        print('Leave existing groups, create new group for extra features.')
                    # Leave any existing groups separate. Any features not 
                    # associated with a group initiate their own group
                    jj = 0
                    for fid,feature in enumerate(features_in_corr):
                        if np.isnan(features_in_corr_group_id[fid]):
                            # Add this feature to new group
                            if jj == 0:
                                # First need to create the new group
                                if verbose:
                                    print('Creating new group '+feature+'----------------------------<===')
                                feature_groups_list.append(feature)
                                jj = 1
                            else:
                                # Add features to the new group
                                feature_groups_list[-1] = feature_groups_list[-1]+','+feature

            # Move to the next highest correlation
            abs_corr[abs_corr==current_highest_corr] = 0
            ii = ii + 1

        # Any remaining features are permuted independently and so
        # by default do not need to be included in the group lists.

        # Concatenate feature_groups_list to feature_groups_str.
        jj = 0
        ff = 0
        ff_list = []
        for group in feature_groups_list:
            if jj == 0:
                feature_groups_str = group
                jj = 1
            else:
                feature_groups_str = feature_groups_str+";"+group

            # Count features and check for duplicates
            for feature in group.split(','):
                ff = ff + 1
                for feature_already_seen in ff_list:
                    if feature == feature_already_seen:
                        print('WARNING: Duplicate feature detected: '+str(feature)+' '+str(ff))
                ff_list.append(feature)

        # Print summary
        print('Started with '+str(len(feature_names_str))+' features.')
        print('Finishing with '+str(ff)+' features in '+str(len(feature_groups_list))+' groups.')

        return feature_groups_str
    
    #===========================================================
    # Step 1: Compute correlations between all inputs and plot. 
    #===========================================================
    
    # With lots of data, this is hard to interpret!
    fig, ax = plt.subplots(figsize=(15,15))
    corr = all_df.corr()
    short_names = [name[:12] for name in corr.columns]
    sns.heatmap(ax=ax, data=np.abs(corr), xticklabels=short_names, yticklabels=short_names, cmap=sns.diverging_palette(220, 10, as_cmap=True,n=3))
    plt.savefig(model_dir+'/sl_all_correlation_heatmap.png')

    group_correlated_features(
        corr,
        corr_cutoff=0.5,
        merge_groups=True,
        onehot_list=['General_Vegetation','River_Gradient','Sediment','Deposition','Hydrogeomorphology'],
        verbose=False
    )

    # Step 2: What is the distribution of correlations?
    # Is there a particular correlation cutoff that is relevant for this data set?
    # In processing corr for plotting, first grab the lower triangle of the correlation
    # heat map since the heat map is symmetric.  Then, reshape it to a vector, and then take
    # the absolute value since we treat negative and positive correlations as the same.
    fig, ax = plt.subplots(figsize=(15,6))
    n, bins, patches = ax.hist(np.reshape(np.tril(np.abs(corr)),-1), 20, density=False, facecolor='g', alpha=0.75, align='mid', histtype='stepfilled')
    plt.savefig(model_dir+'/sl_fpi_correlation_hist.png')

    # Step 3: Which features should be grouped together?
    # Data that are inherently linked (i.e. one-hot and categorical features)?
    # Expert knowledge?
    # Cluster remaining data based on correlation?
    corr_cutoff = 0.5
    hot_spots = corr[np.abs(corr) >= corr_cutoff]
    fig, ax = plt.subplots(figsize=(15,15))
    sns.heatmap(ax=ax, data=np.abs(hot_spots), xticklabels=short_names, yticklabels=short_names, cmap=sns.diverging_palette(220, 10, as_cmap=True,n=3))
    plt.savefig(model_dir+'/sl_fpi_correlation_heatmap.png')    

    #===========================================================
    # Run FPI
    #===========================================================

    # general settings
    job_list = [jobid]
    permute_str = group_correlated_features(
        corr,
        corr_cutoff=0.5,
        merge_groups=True,
        onehot_list=['General_Vegetation','River_Gradient','Sediment','Deposition','Hydrogeomorphology'],
        verbose=False)

    #==========================================================
    # Run FPI for stacked model and each individual submodel
    #==========================================================
    model_fpi_results = list()
    sl_fpi_results = list()

    i = 0
    for job_id in job_list:
        
        print('Loading model for job: '+str(job_id))
        sl = pickle.load(open(model_dir+"/"+"SuperLearners.pkl", "rb"))

        #----------------------------------------------------
        # FPI for stacked model
        #----------------------------------------------------
        model_object = sl[predict_var]
        
        print('FPI on stacked ensemble...')
        result = permute_importance(permute_str, 
                model_object,
                all_df, 
                target_all_df,
                mean_squared_error)
        # Convert back to dataframe, consider using MultiIndex
        # functionality instead of the clunky filter below.
        result_df = pd.DataFrame(result,
                                columns=['Feature',
                                        'Avg_Ratio'+'stack'+str(job_id), 
                                        'Std_Ratio'+'stack'+str(job_id)]).set_index('Feature')
        
        sl_fpi_results.append(result_df)
        
        #----------------------------------------------------
        # FPI for each submodel individually
        #----------------------------------------------------
        for model_name in sl_models:
            model_object = sl[predict_var].named_estimators_[model_name]
            
            print('FPI on ML model: '+model_name+'...')
            result = permute_importance(permute_str, 
                model_object,
                all_df, 
                target_all_df,
                mean_squared_error)
            result_df = pd.DataFrame(result,
                columns=['Feature',
                'Avg_Ratio'+model_name+str(job_id), 
                'Std_Ratio'+model_name+str(job_id)]).set_index('Feature')
            
            model_fpi_results.append(result_df)

    # Merge all dataframes into a single frame with
    # features as the index.
    sl_fpi_results_df = pd.concat(sl_fpi_results,axis=1)
    model_fpi_results_df = pd.concat(model_fpi_results,axis=1)

    # Stacked model results
    print(sl_fpi_results_df.sort_values(by='Avg_Ratiostack'+str(jobid),axis=0,ascending=False))

    # All other model results
    for model in sl_models:
        print('--------'+model+'---------')
        print(model_fpi_results_df['Avg_Ratio'+model+str(jobid)].sort_values(axis=0,ascending=False))
    
    #===========================================================
    # Make heatmap of variable correlations
    #===========================================================
    
    # Working here

    # To view a particular model's list,
    # Choose from ['nusvr', 'mlp', 'ridge', 'xgb']
    # Choose from ['Avg_Ratio', 'Std_Ratio']
    print(pd.DataFrame(
        model_fpi_results_df.filter(
            like='nusvr',axis=1).filter(
            like='Avg_Ratio',axis=1).mean(axis=1)).sort_values(
                by=0,
                axis=0,
                ascending=False))
    
    #===========================================================
    # Save outfile file
    #===========================================================
    
    sl_fpi_results_df.to_csv(f"{model_dir}/sl_fpi_results_df")
    model_fpi_results_df.to_csv(f"{model_dir}/model_fpi_results_df")
    
    #===========================================================
    # Done!
    #===========================================================

