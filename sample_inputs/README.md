# Sample inputs to SuperLearner

Aside from the application code itself in
`../sl_fit_validate`, the two other essential
inputs to the SuperLearner are the
training/testing data and the SuperLearner
configuration.  This directory contains some
examples of each and notes on SuperLearner
configurations.

## SuperLearner configurations

The SuperLearner consists of several "submodels"
whose results are stacked together into a single
result. The submodels are trained independently
and in parallel and then a single "stacking" model
combines them all together.

In the original SuperLearner configuration
(e.g. [Owoyele et al. 2021](https://asmedigitalcollection.asme.org/energyresources/article-abstract/143/8/082305/1103610/An-Automated-Machine-Learning-Genetic-Algorithm)) the stacking
model was a non-negative least squares
regression from Scipy Optimize as in
`superlearner_conf.py` here.  There were
4 submodels.  In the example configurations
here, several additional submodels were
added totalling to 14 submodels.

All submodels are subject to hyperparameters
(in addition to the parameters that are actually fit).
For example, a polynomial model's coefficients
are the parameters while the order of the polynomial
is a hyperparameter. The process of hyperparameter
optimization (HPO) uses the training data to first
optimize the hyperparameters and then the final models
are fit with the best hyperparameters.

Once the SuperLearner is fit, it is saved as a `.pkl`
file and can be opened later for executing additional
predictions.  `.pkl` files are difficult to use since
they must be opened by **exactly** the same version of
the software that created them to ensure correct usage.
(Sometimes you are lucky and this requirement is not
strict across all situations, but it is a risk.)  Also,
`.pkl` files have known security issues which means
we ultimately want to move away from them.

One alternative to `.pkl` is [ONNX](https://onnx.ai/)
format.  ONNX is designed to be a self-contained and
portable machine learning model format. However, ONNX
does not support some of the features in the SuperLearner,
in particular:
1. Scipy's `NNLS` (non-negative least squares) used as the
stacking model and
2. Scikit-Learn's `TransformedTargetRegressor` used to
normalize the targets of the training data. TTR is often
useful because it automatically includes the normalization
of the targets before they are fed into the model and then
automatically applied the inverse transform to get the
un-normalized output from the model.  (I.e. the transform
and its inverse are automatically included in the
fit and prediction pipelines.)

To address the first issue, we use Scikit-Learn's new
(since v0.24) non-negative least squares option in
`LinearRegression`.  To address the second issue, we
simply drop the `TransformedTargetRegressor` from the
SuperLearner pipeline.  TTR appears to be particularly
useful when applying **non-linear** transformations
([Lemaitre, 2022](https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html#sphx-glr-auto-examples-compose-plot-transformed-target-py)) but not so much with linear
transformatins.  The original SuperLearner configuration
uses the `MinMaxScaler` with the TTR, which is a linear
transformation. The exact, best transformation will
vary widely from dataset to dataset and as such removing
the TTR from the SuperLearner still allows users to specify
their own transformations. Users could even experiment
with different transformations by adding additional target
column(s) to their data set as the SuperLearner will fit
independent ensembles of models to each target column.

## Predict data

For the current needs of dynamic-learning-rivers,
we want to keep track of site ID, longitude and 
latitude, but NOT include them as features for
training.  Therefore, the workflow splits the
predict data into `.csv` and `.ixy` files, both
in CSV format.  The first file has 25 features,
the same as the training data.  The second file
is a list of site IDs and sample lons and lats
created from the `.csv` file with:
```bash
 awk '{OFS=","; print NR, NR/10, NR/10}' whondrs_25_inputs_predict.csv > whondrs_25_inputs_predict.ixy
``` 

Future work will streamline/generalize this process 
of keeping track of ancilliary data but not training 
with it.  This is especially useful for running
analytics later and evaluating FPI.
