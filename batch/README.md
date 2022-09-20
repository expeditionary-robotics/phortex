# Code for setting up your workspace
First, clone the `fumes` repository to your home directory on the Engaging cluster. Next, we need to set up the `fumes` Python virutal environemnt. 


## (pipenv) Setup your Python virtual environment on the Engaging cluster 
1. Log into an Engaging login node via ssh, e.g., the following command ssh's (with user geflaspo, assuming your ssh key is located at ~/.ssh/eofe-key) and additionally fowards the port 8888 using Jupyter notebooks on Engaging:
  ```
  ssh -i ~/.ssh/eofe-key -tt geflaspo@eofe8.mit.edu -L 8888:localhost:8888
  ```
2. We're using Python 3.8.3. Add this module to your Engaging node: `module add python/3.8.3`
3. Install pipenv as a user: `pip install --user pipenv`
4. Move into the fumes directory `cd /home/[username]/fumes`
5. Install pipenv `pipenv install` from within the fumes directory 
6. Install the required Python packages: `pipenv install`. This instalation will take a while.
7. You can then activate and deactive the `fumes` virtual environment, using the `pipenv shell` and `exit` commands. These could be, for example, aliased to something easy to remember in your `~/.bashrc` file. 
11. Add the following to your `~/.bashrc` file:
```
alias fumes="cd /home/${USER}/fumes; pipenv shell" 
```


## (virtualenv [depricated]) Setup your Python virtual environment on the Engaging cluster 
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
8. Add the following to your `~/.bashrc` file:
```
alias fumes="cd /home/${USER}/fumes; source /home/${USER}/fumes/bin/activate; source .env" 
```

## Running Python scripts on the cluster
Navigate to the `fumes/batch` directory. If it doesn't already exist, create a `slurm` folder within the `fumes/batch` directory: `mkdir `fumes/batch/slurm`.

Python scripts can be submitted as jobs by running (from the `fumes` top level directory):
```
bash batch/batch_python.sh fumes/examples/post_cruise_examples/demo0-bullseye_flexible_trajectory_opt.py
```
The output and error files will be written the to `slurm` folder within `fumes/batch`. 

NOTE: Always call the `batch_python.sh` script from within the `fumes` top-level directory. Right now, this is the only way that the environment variables get set correctly. TODO: should fix this to let you run it from anywhere. 

The `batch_python.sh` script takes commandline arguments and Python script arguments - see the documentation at the start of the file. For example, I can run:
```
batch/batch_python.sh -m 10 --cores 2 --hours 4 script.py (arg2) (arg3) (arg4)
```

This `batch_python.sh` script is a building block - it allows you to run single python scripts with command-line arguments. This can be adapted to a python script with many parameter values. See the samples in `bulk_batch_predict.sh`. 

## Your ~/.bashrc
Let's add some alias and commands to your ~/.bashrc to make submitting jobs, requesting compute nodes, and checking job statuses easier:
```
# Fumes virtual environment 
alias fumes="cd /home/${USER}/fumes; pipenv shell" 

# Python path
export PYTHONPATH="${PYTHONPATH}:$HOME/fumes"

# Ensure the correct python is used by default
module add python/3.8.3

# Slurm queue and output management
alias qs='qstat -u ${USER}'
alias sq='squeue -u ${USER}'
alias scall='scancel -u ${USER}'
alias rms='rm slurm/*'

# Enable the request of coumpute nodes with specific reuirements
node(){
        if [[ $# -eq 0 ]];
        then
                echo "Node will be available for 12h using 40GB"
                srun -N 1 -n 1 -p newnodes,sched_mit_hill,sched_any --mem=40GB --time=12:00:00 --constraint=centos7 --pty /bin/bash
        elif [[ $# -eq 1 ]]
        then
                echo "Node will be available for ${1}h using 40GB"
                srun -N 1 -n 1 -p newnodes,sched_mit_hill,sched_any --mem=40GB --time=${1}:00:00 --constraint=centos7 --pty /bin/bash
        elif [[ $# -eq 2 ]]
        then
                echo "Node will be available for ${1}h using ${2}GB"
                srun -N 1 -n 1 -p newnodes,sched_mit_hill,sched_any --mem=${2}GB --time=${1}:00:00 --constraint=centos7 --pty /bin/bash
        else
                echo "ERROR: node only accepts two arguments"
                # exit 1
        fi
}
```

## Running commands interactively on the cluster
If you want to run commands interactively on the engaging cluster (instead of submitting them as jobs and checking the status via the `qs` and `sq` commands above), this is also easily done. If you have your `~/.bashrc` setup as described above, from the login node, run on of the following:
```
node # (the default) will connect you to a cluster node with 40GB of memory and 12 hours of time reserved. 
node 7 # will connect you to a cluster node with 40GB of memory for 7 hours of reserved time
node 7 13 # will connect you to a cluster node with 13GB of memory for 7 hours of reserved time
```
One you have been connected to a node (there may be some wait if you request a lot of memory or time, but I usually get allocated one of these interactive nodes quite quickly), navigate to your top-level `fumes` directory and run, e.g.,:
```
pipenv run python fumes/examples/post_cruise_examples/demo2-stationarymtt-iterativeplans.py
```
It is important to use `pipenv run` to run your python code, so that the correct virtual environment is used. You could also just run `pipenv shell` as usual and then use the interactive node as you would normally.


## Other notes
The ssh command above forwards port 8888 from your engaging session to your local computer. If you run a jupyter notebook on the engaging cluster, e.g., `pipenv run jupyter notebook --port 8888`, you should be able to navigate files and run code via a notebook interface in the browser of your computer. It is also possible to connect IDEs, such as VSCode, to the engaging cluster via port 8888, but I have had less success doing this productively. 
