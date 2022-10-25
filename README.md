# sl_core
Core application code for SuperLearner, templates, and documentation.

## Contents

+ `create_conda_env.sh`: Automated build of Conda environment
to run the SuperLearner.

+ `requirements.txt`: All the versions of the software used in
the current Conda environment.  This can be used to build an
environment more quickly; see notes in `create_conda_env.sh`.

+ `sample_inputs`: Some sample inputs for using the SuperLearner;
these files are used by `run.sh` in `sl_fit_validate`.

+ `sample_outputs`: Some sample output files from the SuperLearner.

+ `sl_fit_validate`: Contains the SL `run.sh` and `main.sh`.  The
runscript is a very simple example runscript. **Consider moving
these files into the top level of the repo for combatibility with
`workflow.json`-style workflow deployments.

+ `sl_test`: An old version of the SuperLearner that is kept
here only for development reference/convenience.  It will likely
be removed soon.
