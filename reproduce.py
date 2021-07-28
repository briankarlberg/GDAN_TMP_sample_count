# Main resampling experiment production, 20210724
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

parser = argparse.ArgumentParser(description = 'Cancer cohort and sample size')
parser.add_argument('cohort_name', type = str, help = 'Input TCGA cancer code')
parser.add_argument('sample_size', type = int, help = 'Input sample size')
# Make shell file with sbatch array range specific sample step each cohort
# Print sample counts in notebook to find range - done, array_calculator.ipynb

##### START DEVELOPMENT TOGGLE BLOCKS #####

args = parser.parse_args() # * Devel OFF
cohort_row = best_models[best_models.Cohort == args.cohort_name] # * Devel OFF
sample_size = args.sample_size # * Devel OFF

# command line call: sbatch array_250.sh TCGA_code
# w/ 10-250:10 range for array arg

# cohort_row = best_models[best_models.Cohort == 'BRCA'] # Devel ON
# sample_size = 60 # * Devel ON, put in 250 for BLCA to estimate max run time

date = '2021-07-24' # * script created

##### END DEVELOPMENT TOGGLE BLOCKS #####
# check resampling range prior to deployment on Exacloud

idx = cohort_row.index[0] # position of cohort to map to classifier list

# Process UCSC best model file format
feat_read = best_models.iloc[idx,2].split(
    '_')[1] + '_' + best_models.iloc[idx,2].split(
    '_')[2] + '_' + best_models.iloc[idx,2].split('_')[3]

feat_shard = pd.read_csv('./features/' + feat_read + '.tsv',sep = '\t')

cohort = best_models.iloc[idx,0]
# debug 07-21, where does cohort go?

rpt_fld_file = pd.read_csv(
    './cross_val/' + cohort + '_CVfolds_5FOLD_v12_20210228.tsv',
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
 [ DecisionTreeClassifier(criterion = 'entropy', max_depth = 8,
                          min_samples_split = 3) ],
 [ LogisticRegression(C=10,max_iter = 500,solver = 'lbfgs') ],
 [ RandomForestClassifier(criterion = 'entropy',n_estimators = 200) ],
 [ GaussianNB(var_smoothing = 1e-06) ],
 [ ExtraTreesClassifier(criterion = 'gini',n_estimators = 128) ],
 [ RandomForestClassifier(criterion = 'entropy',n_estimators = 200) ]
]

clf = clf_lst[idx][0] # Initialize classifier for cohort

def extrct_nms(shrd_pth):
    slash_split = shrd_pth.split('/')
    uniq = slash_split[-1].split('.')[0]
    mthd_dtyp = uniq.split('_')[1] + '_' + uniq.split('_')[2]
    return (mthd_dtyp)

(mthd_dtyp) = extrct_nms('./features/'+feat_read+'.tsv') # get column name

results_header = ['Sample_ID',
                  'Repeat',
                  'Fold',
                  'Test',
                  'Label',
                  str(clf) + '|' + mthd_dtyp + '|' + date + '|c',
                  'Resampling']

results_storeDF = pd.DataFrame(columns = results_header) # Create results Frame

checkpoint_time  = time.time()
     ####### TOGGLE for devel, production use list(range(1,101)) #####
for resampling in list(range(1, 101)): # Sampling loop CHECK
    print('Resampling: ' + str(resampling))
    shard_subset = feat_shard.sample(sample_size) # Select a random sample

    if resampling in list(range(10,101,10)):
        ten_resamplings_time = time.time() - checkpoint_time

        progress_meter = pd.DataFrame(
            data = {'Resampling_number': resampling},
                                             index = [0])
        progress_meter.to_csv( ##### Check this directory for deployment #####
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
            transfer_row = pd.DataFrame(trainDF[-1:].values, index=[0],
                                        columns=trainDF.columns)
            trainDF.drop(trainDF.tail(1).index,inplace=True)
            transfer_row.iloc[0,-1] = 1
            testDF = testDF.append(transfer_row)
            continue

        while trainDF.Labels.nunique() == 1: # Error two, only one subtype in training set
            shard_subset = feat_shard.sample(sample_size)
            trainDF = shard_subset[shard_subset['rpt_fld'] == 0]
            testDF = shard_subset[shard_subset['rpt_fld'] == 1]

        if len(testDF) == 0: # Error 1.2, no samples in test again after resampling
            transfer_row = pd.DataFrame(trainDF[-1:].values, index=[0],
                                        columns=trainDF.columns)
            trainDF.drop(trainDF.tail(1).index,inplace=True)
            transfer_row.iloc[0,-1] = 1
            testDF = testDF.append(transfer_row)

        if trainDF.Labels.nunique() == 1:
            print('continue')
            continue

        X_trn = trainDF.drop(columns = [cohort,'Labels','rpt_fld'])
        x_tst = testDF.drop(columns = [cohort,'Labels','rpt_fld'])
        y_trn = trainDF['Labels'].str.split('_',expand=True)[1].astype(int)
        y_tst = testDF['Labels'].str.split('_',expand=True)[1].astype(int)

        trnD = {'Sample_ID':trainDF[cohort], # Build storage frame for training samples
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

        y_prd_trn = clf.predict(X_trn) # Predict on training sample features
        y_prd_tst = clf.predict(x_tst) # Predict on testing sample features
        trn_series = pd.Series(y_prd_trn.astype(str)) # convert to series
        tst_series = pd.Series(y_prd_tst.astype(str)) # convert to series
        trn_labl = cohort + '_' + trn_series.str[0] # reattach cancer label
        tst_labl = cohort + '_' + tst_series.str[0] # reattach cancer label

         # Synapse format plus resampling column
        rsltsTRN[str(clf) + '|' + mthd_dtyp + '|' + date + '|c'] = trn_labl
        rsltsTST[str(clf) + '|' + mthd_dtyp + '|' + date + '|c'] = tst_labl
        rsltsTRN['Resampling'] = resampling # For sort to fit indv curves
        rsltsTST['Resampling'] = resampling
        results_storeDF = pd.concat([results_storeDF, rsltsTRN, rsltsTST])

file_time =  time.time() - file_start_time

##### CHECK path and directory for deployment #####
results_storeDF.to_csv(
            './results/' +
            cohort + '.'+ date +
            '.sample_size.' + str(sample_size).zfill(3) +
            '.runtime.' + str(round(file_time)).zfill(5) +
            '.tsv',
            index = None, sep = '\t')
