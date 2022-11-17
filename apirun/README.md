# Launching sl_core with the PW API

The scripts in this directory are an example for how to launch a PW workflow
from another computer via a PW account's API key. The script `main.sh` contains
all the relevant command line launch options including the command to launch
the API request and all the options required by the workflow itself.

Please note that in order for `main.sh` to run, the user needs to specify
three environment variables in advance:
1. PARSL_CLIENT_HOST: the address of the PW platform, e.g. `cloud.parallel.works`,
2. PW_API_KEY: the API key associated with a PW user account, and
3. PW_USER: the PW user account associated with the API key, above.

The API key must be treated with the same level of care as a password.
