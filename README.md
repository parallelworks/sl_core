# sl_core
Core application code for SuperLearner, templates, and documentation.

## TL;DR

```bash
# Run the SuperLearner locally with sample data
git clone https://github.com/parallelworks/sl_core
cd sl_core
./local_superlearner_test.sh
```

## Usage

For testing the usage of the SuperLearner, please start with
`train_predict_evaluate.sh` which is a wrapper that launches
the central part of the SuperLearner code `main.py` as well as
the two post-training operations of making predictions and
estimating errors, `predict.py` and `errors.py`, respectively. 
The launch script `local_superlearner_test.sh` is a template for
how to specify the options of `train_predict_evaluate.sh`.

Broadly, `train_predict_evaluate.sh` is meant to be launched as
part of a larger workflow that includes syncing machine learning
code, data, and archives via GitHub and running multiple instances
of the SuperLearner in parallel.  This workflow is defined by
`workflow.sh` (the code) and `workflow.xml` which is a "form"
displayed by the PW platform that allows for users to specify
workflow parameters and launch the workflow via calls to the PW
API.  API-launching a workflow is particularly useful when
integrating the workflow with GitHub actions since the actions
run as Docker containers on GitHub and can be set up to launch
PW workflow through the API.  Please see the
[ML-archive repository](https://github.com/parallelworks/dynamic-learning-rivers) 
associated with this workflow for more information.

## Install

There are three ways to install a workflow on Parallel Works.
1. **Use a `github.json` if available.** A GitHub-integrated
workflow can be automatically cloned to the PW platform if the
user has access to the repository.  For example, the JSON code
block below would work as the contents of `github.json` for
this workflow because it points to this workflow's public
repository. No other files are needed on the PW platform since
the rest of the necessary files are cloned each time the user
selects the workflow.
```
{
    "repo": "https://github.com/parallelworks/sl_core.git",
    "branch": "main",
    "dir": ".",
    "xml": "workflow.xml",
    "thumbnail": "superlearner.png",
    "readme": "README.md"
}
```

2. **Copy the workflow from the PW Marketplace if available.**
Workflows (GitHub-integrated or "classic", see below) can be
shared with other PW users on the PW Marketplace.  To get to
the Marketplace, click on PW account's name in the upper
right corner and then select Marketplace (globe icon) from the
drop down menu.

3. **Install the workflow by copying files.** The "classical"
way to install a PW workflow is to first create a new workflow;
in the case of `sl_core`, use a `Bash` type workflow. Then, in
the PW IDE terminal:
```bash
# Navigate to the workflow directory created by the platform.
cd /pw/workflows/sl_core

# Remove the default files in this directory.
rm -f

# Manually copy the workflow files into the workflow directory.
git clone https://github.com/parallelworks/sl_core .
```
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
