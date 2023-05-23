# Notes on SuperLearner stages

## Train

The training stage uses the `superlearner_conf.py` (or other file specified by the user)
along with input files and input parameters to train the machine learning models specified
in the main configuration file.  The default configuration is a `scikit-learn` 
`StackedEnsembleRegressor` that trains several "sub-models" and then assigns weights to
the best performing sub-models to generate an overall "blended-model".

## Predict

## PCA

In an effort to estimate errors, Principal component analysis (PCA) is used to determine which
of the data we are predicting on is most different from the rest of the data set. A second
error metric is built from using the value of each target as an estimator for the potential
prediction error.

## FPI

Which of the many possible inputs (i.e. "features") is most important? Here, we address this 
question by permuting the inputs to see which have the biggest impact on the results. We also 
need to experiment with grouping together correlated features. If features are significantly 
correlated with each other, then they need to be permuted together, otherwise when a feature 
is permuted, the model can still get information from the other, correlated features, meaning 
that the final predicted values may not change substantially -> the importance will be 
"diluted" across the correlated features.

Although no model fitting is required, this can take quite a few minutes to run for a small 
data set - bigger data sets and exploring variability among models may require a separate 
parallelized workflow. For now, we always run FPI but in the future an FPI toggle switch
in the workflow form will likely be needed.