### Sample size effect on accuracy experiment.  

Machine learning model prediction scores are a function of both the predictive signal within the data and the number of training of samples available. This experiment disentangles the sample size effect by sub-sampling raw data in incremental steps to generate learning curves. Different combinations of resampling rates and cross validation repeats were explored; the main experiment used 100 resamplings with 10 cross folds per sample step size. This was parallelized on OHSU's Exacloud cluster. The application is 26 cancers within The Cancer Genome Atlas. The result is predictions of maximum F1 scores for classifying subtypes within the 11 primary tumor types comprising of less than 250 samples. The error associated with these predictions is inferred from developing the prediction method on the cancers with at least 250 samples.  

[insert manuscript url].  

[insert link to poster].  

original operational notes, 2021-11-17:  

Last unmerged plotting branch is "inverse power law". 

sample_response_DF_20210805.tsv

is input object for Figure 7

from:

confidence_interval.ipynb
