# step_input_substep.py
import time
start_time = time.time()

from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import f1_score
import pandas as pd
import math
import statistics

import argparse
parser = argparse.ArgumentParser(description = 'Cancer and sample size input')
parser.add_argument('cancer_name', type = str, help='Input cancer name')
parser.add_argument('sample_size', type = int, help='Input sample size')

parser.add_argument('sub_step', type = int, help='Input substep number') # Most rececnt addition

args = parser.parse_args() # * Turn off for devel

three_col = pd.read_csv('sk_grid_best_models.tsv', sep = '\t')

date = '2021-06-15' # * script created

# cancer_row = three_col[three_col.Cancer == 'ACC'] # devel mode
cancer_row = three_col[three_col.Cancer == args.cancer_name] # * off for devel

# set_size = 20 # list(range(10,101,10)) # devel mode
set_size = args.sample_size # * devel off, command line input 2

# sub_step = 1 # devel mode
sub_step = args.sub_step # * devel off, e.g. sbatch array_sub.sh call w/ 1-10:1 range for file name

idx = cancer_row.index[0]

# Process UCSC best model file format
feat_read = three_col.iloc[idx,2].split('_')[1]+'_'+three_col.iloc[idx,2].split('_')[2]+'_'+three_col.iloc[idx,2].split('_')[3]

feat_shard = pd.read_csv('./features/'+feat_read+'.tsv',sep='\t')

cohort = three_col.iloc[idx,0]

rpt_fld_file = pd.read_csv('./cross_val/'+cohort+'_CVfolds_5FOLD_v12_20210228.tsv',
                                                        sep='\t')
clf_list = [
    [SGDClassifier(alpha=0.001,loss='perceptron',penalty='l1')],
    [SVC(C=0.2,kernel='linear')],
    [SVC(C=1,kernel='linear')],
    [SVC(C=2,kernel='linear')],
    [SVC(C=1.5,kernel='linear')],
    [SVC(C=1,kernel='linear')],
    [SVC(C=1.5,kernel='rbf')],
    [SVC(C=2,kernel='linear')],
    [SGDClassifier(alpha=0.0001,loss='squared_hinge',penalty='l1')],
    [SVC(C=0.2,kernel='linear')],
    [SVC(C=2,kernel='linear')],
    [SVC(C=0.5,kernel='linear')],
    [SVC(C=0.2,kernel='linear')],
    [SVC(C=1.5,kernel='linear')],
    [SVC(C=0.2,kernel='linear')],
    [SVC(C=0.2,kernel='linear')],
    [PassiveAggressiveClassifier(C=0.8)],
    [PassiveAggressiveClassifier(C=1.0)],
    [PassiveAggressiveClassifier(C=0.4)],
    [SVC(C=0.2,kernel='linear')],
    [SGDClassifier(alpha=0.1,loss='hinge',penalty='l2')],
    [SVC(C=2,kernel='poly')],
    [SVC(C=1.5,kernel='rbf')],
    [SVC(C=1.5,kernel='linear')],
    [SVC(C=0.2,kernel='rbf')],
    [SVC(C=0.2,kernel='linear')],
]

clf = clf_list[idx][0] # Initialize classifier for cohort

def extrct_nms(shrd_pth):
    slash_split = shrd_pth.split('/')
    cncr = slash_split[-1].split('_')[0]
    uniq = slash_split[-1].split('.')[0]
    mthd_dtyp = uniq.split('_')[1]+'_'+uniq.split('_')[2]
    return (cncr, mthd_dtyp)

(cncr,mthd_dtyp) = extrct_nms('./features/'+feat_read+'.tsv') # get column name components

rate_scores = [] # Storage list for scores at different sample counts
errors = []
sample_counts = []

# sample_counts.append(set_size) # arg parse from list in shell file
hdr_lst=['Sample_ID', 'Repeat', 'Fold', 'Test', 'Label']
unique_name=str(clf)+'|'+mthd_dtyp+'|'+date+'|c'
results_frame_header = hdr_lst.append(unique_name)
results_storeDF=pd.DataFrame(columns=results_frame_header) # Empty data frame with columns for this classifier

# for set_size in set_list: # * Devel loop, replaced with array arg from shell script
if set_size >= len(feat_shard):
    exit()
#     print(set_size)

loop_start_time = time.time()
for i in list(range(0,10)): # Ten resamplings
    sample_counts.append(set_size)
#     print(i)

    shard_subset = feat_shard.sample(set_size)

    rpt_scores = [] # scores for error
    for rx_fx in rpt_fld_file.iloc[:,2:12]:
        from sklearn.linear_model import SGDClassifier
        clf = clf_list[idx][0]


        feat_shard['rpt_fld'] = rpt_fld_file[rx_fx] #label molecular shard for train/test split, will overwrite
        labeled_subset = feat_shard.iloc[shard_subset.index,:]

        trainDF = labeled_subset[labeled_subset['rpt_fld']==0]
        testDF = labeled_subset[labeled_subset['rpt_fld']==1]

        if len(testDF) == 0: # Error one, no samples in test set
            transfer_row = pd.DataFrame(trainDF[-1:].values, index=[0], columns=trainDF.columns)
            trainDF.drop(trainDF.tail(1).index,inplace=True)
            transfer_row.iloc[0,-1] = 1
            testDF = testDF.append(transfer_row)
#                 print('zero_test')
            continue

        while trainDF.Labels.nunique() == 1: # Error two, only one subtype in training set
            shard_subset = feat_shard.sample(set_size)
            trainDF = shard_subset[shard_subset['rpt_fld']==0]
            testDF = shard_subset[shard_subset['rpt_fld']==1]
#                 print('one_label')

        X_trn=trainDF.drop(columns=[cncr,'Labels','rpt_fld']) # raw feature columns remain
        x_tst=testDF.drop(columns=[cncr,'Labels','rpt_fld'])
        y_trn=trainDF['Labels'].str.split('_',expand=True)[1].astype(int) # Strip TCGA code off number
        y_tst=testDF['Labels'].str.split('_',expand=True)[1].astype(int)

        trnD={'Sample_ID':trainDF[cncr], # Build storage frame with info for training samples, will attach prediction results column to RHS
          'Repeat':rx_fx.split(':')[0],
          'Fold':rx_fx.split(':')[1],
          'Test':trainDF.rpt_fld,
          'Label':trainDF.Labels}
        rsltsTRN=pd.DataFrame(data=trnD)
        rsltsTRN.reset_index(inplace=True,drop=True)

        tstD={'Sample_ID':testDF[cncr],
          'Repeat':rx_fx.split(':')[0],
          'Fold':rx_fx.split(':')[1],
          'Test':testDF.rpt_fld,
          'Label':testDF.Labels}
        rsltsTST=pd.DataFrame(data=tstD)
        rsltsTST.reset_index(inplace=True,drop=True)

        clf.fit(X_trn, y_trn)

        y_prd_trn = clf.predict(X_trn) # Predict on training sample feature values
        y_prd_tst = clf.predict(x_tst) # Predict on testing sample feature values

        rpt_score=f1_score(y_tst,y_prd_tst,average='weighted')
        rpt_scores.append(rpt_score) # Error bars data

        trn_series = pd.Series(y_prd_trn.astype(str)) # convert prediction output array to series; now has index
        tst_series = pd.Series(y_prd_tst.astype(str))

        trn_labl=cncr+'_'+trn_series.str[0] # reattach cancer label to predictions
        tst_labl=cncr+'_'+tst_series.str[0]

         # Make a column header name for classifier plus feature set which is "the model" in Synapse format

        rsltsTRN[unique_name]=trn_labl
        rsltsTST[unique_name]=tst_labl

        results_storeDF=pd.concat([results_storeDF,rsltsTRN,rsltsTST]) # Stack up the prediction results for this repeat fold

    stdv = statistics.stdev(rpt_scores)
    errors.append(stdv)

    test_set=results_storeDF[results_storeDF.Test==1]
#     print(len(test_set)) # running average
    y_true_str=test_set.Label
    y_true=[]
    for k in y_true_str:
        splt=k.split('_')
        y_str_ele=splt[1]
        y_int_ele=int(y_str_ele)
        y_true.append(y_int_ele)
#     print(len())
    col=test_set.iloc[:,5]
    y_pred=[]
    for j in col:
        splt=j.split('_')
        y_str_ele=splt[1]
        y_int_ele=int(y_str_ele)
        y_pred.append(y_int_ele)
#     print(len(y_pred))
    score=f1_score(y_true,y_pred,average='weighted')
    rate_scores.append(round(score, 3))
row_df = pd.DataFrame([[cncr, sample_counts, rate_scores, errors]], columns=['Cancer', 'Sample_counts', 'Rate_scores', 'Error'])
# write to out_file
loop_end_time = time.time()
file_time =  loop_end_time - loop_start_time
row_df.to_csv('./results/test/'+ # write results file for this classfier, cohort, sample step combo, date
             cncr+'_'+date+'_set.'+str(set_size)+'.'+str(sub_step)+'_runtime:'+str(round(file_time))+'.tsv',
                      index=None, sep='\t')