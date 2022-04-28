#!/bin/bash
# Submits a given Python script to the cluster using sbatch
#
# Example uses:
#   batch/batch_python.sh script.py (arg2) (arg3) (arg4)
#   batch/batch_python.sh --memory 10 script.py (arg2) (arg3) (arg4)
#   batch/batch_python.sh -m 10 --cores 2 --hours 4 script.py (arg2) (arg3) (arg4)
#
# Named args (optional, should precede Python script name to override default)
#   -m | --memory : memory to request in GB
#   -c | --cores : number of cores to request
#   -h | --hours : number of hours to request
#   -mn | --minutes: number of minutes to request
#   -e | --environment : conda environment to load
#   -d | --dependency : dependency list for jobs
#   -g | --gpus : number of gpus to request
#   --mail : whether to email user; ALL, BEGIN, END, FAIL or NONE
#   --mail-user : list of comma separated email addresses to receive email
#     (by default, will email the user executing this script)
#
# Additional args
#   script.py : name of Python script followed by any of its arguments

MEMORY=50
CORES=8
HOURS=12
MINUTES=0
GPUS=0
ENVIRONMENT=fumes
MAIL=ALL
MAIL_USER=${USER}@mit.edu
DEPENDENCY=''

# Assign named arguments to variables (and shift them)
while [ "$1" != "" ]; do
  case $1 in
    -m | --memory )         shift
      MEMORY=$1
      ;;
    -c | --cores )    		shift
      CORES=$1
      ;;
    -h | --hours )    	    shift
      HOURS=$1
      ;;
    -mn | --minutes )    shift
      MINUTES=$1
      ;;
    -e | --environment )    shift
      ENVIRONMENT=$1
      ;;
    -d | --dependency )    shift
      DEPENDENCY=$1
      ;;
    -g | --gpus )    shift
      GPUS=$1
      ;;
    --mail )				shift
      MAIL=$1
      ;;
    --mail-user )				shift
      MAIL_USER=$1
      ;;
    * )                     break
  esac
  shift
done

echo -e "\nJob will run with ${MEMORY}GB, $CORES cores, and dependency $DEPENDENCY for ${HOURS}h, ${MINUTES}min."

# Concatenate remaining arguments to be used in sbatch script
# e.g., args="script.py arg2 arg3 arg4"
args="$*"
# Keep track of arguments with spaces replaced by underscores
# e.g., args_underscore="script.py_arg2_arg3_arg4"
args_underscore="${args// /_}" 

# MIT users do not have access to sched_engaging_default nodes
if [ ${USER} = "geflaspo" ]; then
    NODE_LIST="newnodes,sched_mit_hill,sched_any"
elif [ ${USER} = "vpreston" ]; then
    # Note: sched_mit_darwin2 for long runs available with this user
    NODE_LIST="newnodes,sched_any,sched_mit_hill"
else
    NODE_LIST="newnodes,sched_mit_hill,sched_engaging_default,sched_any"
fi

echo "Running: pipenv run python $args"

sbatch <<-EOT
#!/bin/bash
#################
#set a job name
#SBATCH --job-name=$args_underscore
#################
#only use machines with CentOS 7 operating system
#SBATCH --constraint=centos7
#################
#set output file names; %j will be replaced with job ID
#SBATCH --output=batch/slurm/%j.out
#SBATCH --error=batch/slurm/%j.err
#################
#set queue (default time is 12 hours)
#SBATCH -p ${NODE_LIST}
#################
#number of nodes, cores per node and memory
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=$CORES
#SBATCH --mem=${MEMORY}gb
#SBATCH --gres=gpu:${GPUS}
#################
#time before script is killed
#SBATCH -t ${HOURS}:${MINUTES}:00
#################
#set dependencies
#SBATCH --dependency=${DEPENDENCY}
#################
#get emailed about job BEGIN, END, and FAIL (or ALL)
#SBATCH --mail-type=${MAIL}
#SBATCH --mail-user=${MAIL_USER}
#################
# Print command
echo -e "pipenv run python $args\n"
# Load supporting modules
module add python/3.8.3
# Execute python script
pipenv run python $args
EOT
