# FUMES Planner
For intermittent deployment missions, this planner enables global mission planning and localized, adaptive optimization from observations. The base of the `fumes` planner is a trajetory optimizer, that uses black-box optimization methods to select trajectory parameters the optimize a given reward function.
![Alt Text](traj_single.gif)

 The optimizer can also chain together a given trajectory type to target dynamic phenomnea.
![Alt Text](traj_chainer.gif)

Project Python dependencies are managed by `pipenv`. You may need to install `pipenv` on your machine. To generate your `fumes` virtual environment with all dependencies installed, navigate to `src/planning/fumes` on your local machine and run the command: `pipenv install`. Then, to activate your environment, run `pipenv shell` from the same directory. To install new dependencies to the virtual environment, run `pipenv install <package_name>`.

You may want to add the following to your `~/.bashrc` (or appropriate file):
```
alias fumes="cd /<path to rrg repo>/rrg/src/planning/fumes; pipenv shell"
```
Then, when starting a new terminal, typing `fumes` will go to the correct directory, activate the fumes Python virtual environment, and load the necessary environment variables, located in `.env`.

The `.env` file sets the environment variables `FUMES_DATA` and `FUMES_OUTPUT`, which is the path to data and outputs from robot simulations respectively.  Data files supporting the `fumes` code (simulation models, etc.) should be places in `fumes/data` and output files from, e.g., robot simulation runs, will be saved in `fumes/output`.

Example robot simulations can be found as Python scripts in `fumes/fumes/examples`.

Test cases are found in `fumes/fumes/tests` and can be run with `pytest`, e.g. `pytest fumes/tests`.