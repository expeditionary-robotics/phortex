# Code for setting up your workspace
First, clone the `fumes` repository to your home directory on the Engaging cluster. Next, we need to set up the `fumes` Python virutal environemnt. 

## Setup your Python virtual environment on the Engaging cluster 
We're following the instructions [here](https://engaging-web.mit.edu/eofe-wiki/virtual_envs/scripted/python_venv/) to setup a Python virtual environment on Engaging. 

1. Log into an Engaging login node via ssh, e.g., the following command ssh's and additionally fowards the port 8888 using Jupyter notebooks on Engaging:
  ```
  ssh -i ~/.ssh/eofe-key -tt geflaspo@eofe8.mit.edu -L 8888:localhost:8888
  ```
2. We're using Python 3.8.3. Add this module to your Engaging node: `module add python/3.8.3`
3. Create a `fumes` environment in your home directory: `python3 -m venv fumes`
4. Activate your environment: `source /home/[username]/fumes/bin/activate`
5. Change into/ensure that you are in the fumes repo: `cd fumes`
6. Install the required Python packages: `pip install -r requirements.txt`. This instalation will take a while. Note: if the `requirements.txt` file needs to be updated to match an updated `Pipfile`, then run `pipenv lock -r > requirements.txt` from within the `fumes` directory. 
7. You can then activate and deactive the `fumes` virtual environment, using the above activate command and the `deactivate` command. These could be, for example, aliased to something easy to remember in your `~/.bashrc` file. 
