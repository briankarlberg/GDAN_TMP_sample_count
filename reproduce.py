# reproduce.py
# Put on exacloud and call with shell script modified to sample range
import time
file_start_time = time.time()
import pandas as pd
import argparse

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

best_models = pd.read_csv( # replaced three_col
    'skgrid_best_models_20210710_reformatted.tsv',
    sep = '\t')
best_models.rename(columns={"Cancer": "Cohort"},inplace = True)

parser = argparse.ArgumentParser(description = 'Cancer and sample size input')
parser.add_argument('cohort_name', type = str, help = 'Input TCGA cancer code') # Arg one in shell script
parser.add_argument('sample_size', type = int, help = 'Input sample size') # Arg two in shell script
# Make shell file with sbatch array range specific sample step for each cohort with correct sample range e.g. #SBATCH -a 10-250:10
# Print sample counts in notebook to find range

args = parser.parse_args() # * Devel OFF
cohort_row = best_models[best_models.Cohort == args.cohort_name] # * Devel OFF
sample_size = args.sample_size # * Devel OFF, e.g. sbatch array_250.sh call w/ 10-250:10 range
# Local test strategy: call with step size 10

# cohort_row = best_models[best_models.Cohort == 'ACC'] # Devel ON
# sample_size_list = list(range(10,71,10)) # * Devel ON, put in 250 for LIHCCHOL, BRCA, etc to estimate max run time

date = '2021-07-21' # * script created

idx = cohort_row.index[0] # Get index position of cohort to map to classifier list

# Process UCSC best model file format
feat_read = best_models.iloc[idx,2].split(
    '_')[1] + '_' + best_models.iloc[idx,2].split(
    '_')[2] + '_' + best_models.iloc[idx,2].split('_')[3]

feat_shard = pd.read_csv('./features/' + feat_read + '.tsv',sep = '\t')

cohort = best_models.iloc[idx,0]
# debug 07-21, where does cohort go?

rpt_fld_file = pd.read_csv('./cross_val/' + cohort + '_CVfolds_5FOLD_v12_20210228.tsv',
                                                        sep = '\t')
clf_lst = [
 [ ExtraTreesClassifier(criterion = 'gini',n_estimators = 128) ],
 [ LogisticRegression(C = 100,max_iter = 500,solver = 'newton-cg') ],
 [ RandomForestClassifier(criterion = 'entropy',n_estimators = 150) ],
 [ LogisticRegression(C = 100,max_iter = 500,solver = 'newton-cg') ],
 [ ExtraTreesClassifier(criterion = 'entropy',n_estimators = 128) ],
 [ ExtraTreesClassifier(criterion = 'gini',n_estimators = 128) ],
 [ LogisticRegression(C = 10,max_iter = 500,solver = 'newton-cg') ],
 [ LogisticRegression(C = 1.0,max_iter = 500,solver = 'lbfgs') ],
 [ SGDClassifier(alpha = 0.001,loss = 'modified_huber',penalty = 'l1') ],
 [ RandomForestClassifier(criterion = 'entropy',n_estimators = 150) ],
 [ LogisticRegression(C = 100,max_iter = 500,solver = 'newton-cg') ],
 [ ExtraTreesClassifier(criterion='entropy',n_estimators = 128) ],
 [ LogisticRegression(C = 1.0,max_iter = 500,solver = 'liblinear') ],
 [ LogisticRegression(C = 100,max_iter = 500,solver = 'lbfgs') ],
 [ ExtraTreesClassifier(criterion = 'gini',n_estimators = 96) ],
 [ LogisticRegression(C = 0.01,max_iter = 500,solver = 'newton-cg') ],
 [ LogisticRegression(C = 100,max_iter = 500,solver = 'lbfgs') ],
 [ RandomForestClassifier(criterion = 'gini',n_estimators = 120) ],
 [ RandomForestClassifier(criterion = 'entropy',n_estimators = 80) ],
 [ SVC(C = 1,kernel = 'linear') ],
 [ DecisionTreeClassifier(criterion = 'entropy',max_depth = 8,min_samples_split = 3) ],
 [ LogisticRegression(C=10,max_iter = 500,solver = 'lbfgs') ],
 [ RandomForestClassifier(criterion = 'entropy',n_estimators = 200) ],
 [ GaussianNB(var_smoothing = 1e-06) ],
 [ ExtraTreesClassifier(criterion = 'gini',n_estimators = 128) ],
 [ RandomForestClassifier(criterion = 'entropy',n_estimators = 200) ]
]

clf = clf_lst[idx][0] # Initialize classifier for cohort based on position in best_model file

def extrct_nms(shrd_pth):
    slash_split = shrd_pth.split('/')
    # cncr = slash_split[-1].split('_')[0] # For delete, replaced with cohort
    uniq = slash_split[-1].split('.')[0]
    mthd_dtyp = uniq.split('_')[1] + '_' + uniq.split('_')[2]
    # return (cncr, mthd_dtyp) # For delete, replaced with cohort
    return (mthd_dtyp)

#(cncr,mthd_dtyp) = extrct_nms('./features/'+feat_read+'.tsv') # For delete
(mthd_dtyp) = extrct_nms('./features/'+feat_read+'.tsv') # get column name components
# debug 07-21, is this different than cohort?

results_header = ['Sample_ID', 'Repeat', 'Fold', 'Test', 'Label']
unique_name = str(clf) + '|' + mthd_dtyp + '|' + date + '|c'
full_results_header = results_header.append(unique_name)
results_storeDF = pd.DataFrame(columns = full_results_header) # Create results DataFrame
            ####### TOGGLE for devel, production use list(range(1,101)) #####
checkpoint_time  = time.time()
for resampling in list(range(1, 101)): # Sampling loop
#        print('Resampling: ' + str(resampling))
    shard_subset = feat_shard.sample(sample_size) # Select a random sample

    if resampling in list(range(10,101,10)):

        ten_resamplings_time = time.time() - checkpoint_time

        progress_meter = pd.DataFrame(
            data = {'Resampling_number': resampling},
                                             index = [0])
        progress_meter.to_csv(
            './progress_meter/' + cohort +
            '.sample_size.' + str(sample_size).zfill(3) +
            '.resampling.' + str(resampling) +
            '.time.' + str(round(ten_resamplings_time)) +
            '.tsv',
            sep = '\t')

        checkpoint_time  = time.time()
    for rx_fx in rpt_fld_file.iloc[:,2:12]: # Cross validaton loop

        feat_shard['rpt_fld'] = rpt_fld_file[rx_fx]
        labeled_subset = feat_shard.iloc[shard_subset.index,:]

        trainDF = labeled_subset[labeled_subset['rpt_fld'] == 0]
        testDF = labeled_subset[labeled_subset['rpt_fld'] == 1]

        if len(testDF) == 0: # Error one, no samples in test set
            transfer_row = pd.DataFrame(trainDF[-1:].values, index=[0], columns=trainDF.columns)
            trainDF.drop(trainDF.tail(1).index,inplace=True)
            transfer_row.iloc[0,-1] = 1
            testDF = testDF.append(transfer_row)
#                 print('zero_test')
            continue

        while trainDF.Labels.nunique() == 1: # Error two, only one subtype in training set
            shard_subset = feat_shard.sample(sample_size)
            trainDF = shard_subset[shard_subset['rpt_fld'] == 0]
            testDF = shard_subset[shard_subset['rpt_fld'] == 1]
#                 print('one_label')

        if len(testDF) == 0: # Error 1.2, no samples in test again after resampling
            transfer_row = pd.DataFrame(trainDF[-1:].values, index=[0], columns=trainDF.columns)
            trainDF.drop(trainDF.tail(1).index,inplace=True)
            transfer_row.iloc[0,-1] = 1
            testDF = testDF.append(transfer_row)

        if trainDF.Labels.nunique() == 1:
            print('continue')
            continue

        X_trn = trainDF.drop(columns = [cohort,'Labels','rpt_fld']) # raw feature columns remain
        x_tst = testDF.drop(columns = [cohort,'Labels','rpt_fld'])
        y_trn = trainDF['Labels'].str.split('_',expand=True)[1].astype(int) # Strip TCGA code off number
        y_tst = testDF['Labels'].str.split('_',expand=True)[1].astype(int)

        trnD = {'Sample_ID':trainDF[cohort], # Build storage frame with info for training samples
          'Repeat':rx_fx.split(':')[0],    # prediction results column attaches to RHS
          'Fold':rx_fx.split(':')[1],
          'Test':trainDF.rpt_fld,
          'Label':trainDF.Labels}
        rsltsTRN = pd.DataFrame(data = trnD)
        rsltsTRN.reset_index(inplace = True, drop = True)

        tstD = {'Sample_ID':testDF[cohort],
          'Repeat':rx_fx.split(':')[0],
          'Fold':rx_fx.split(':')[1],
          'Test':testDF.rpt_fld,
          'Label':testDF.Labels}
        rsltsTST = pd.DataFrame(data = tstD)
        rsltsTST.reset_index(inplace = True, drop = True)

        clf.fit(X_trn, y_trn)

        y_prd_trn = clf.predict(X_trn) # Predict on training sample feature values
        y_prd_tst = clf.predict(x_tst) # Predict on testing sample feature values
        trn_series = pd.Series(y_prd_trn.astype(str)) # convert prediction output array to series; now has index
        tst_series = pd.Series(y_prd_tst.astype(str)) # convert prediction output array to series; now has index
        trn_labl = cohort + '_' + trn_series.str[0] # reattach cancer label to predictions
        tst_labl = cohort + '_' + tst_series.str[0] # reattach cancer label to predictions

         # Make a column header name for classifier plus feature set which is "the model" in Synapse format
        rsltsTRN[unique_name]=trn_labl
        rsltsTST[unique_name]=tst_labl
        results_storeDF = pd.concat([results_storeDF, rsltsTRN, rsltsTST]) # Stack up the prediction results

file_time =  time.time() - file_start_time
# Add step iteration to file naming
results_storeDF.to_csv(
            './results/' +
            cohort + '.'+ date +
            '.sample_size.' + str(sample_size).zfill(3) +
            '.runtime.' + str(round(file_time)).zfill(5) +
            '.tsv',
            index = None, sep = '\t')