#!/bin/bash -x
# Submits batch predict with various parameter configurations to the cluster
#
# Usage:
#   bulk_batch_predict.sh  <SOME_PARAM>
#
# Args:
#   <SOME_PARAM>: some description 

param=${1:-std_ens}
cmd_prefix="batch_python.sh --memory 3 --cores 3 --hours 2 --minutes 15"
script="../fumes/examples/post_cruise_examples/demo0-bullseye_flexible_trajectory_opt.py"

# List of parameter settings to try
experiment_list=("None" "current_w" "uniform" "prev_y" "trend_y" "mean_y" "prev_g" "mean_g" "doy" "adaptive")
for param in ${experiment_list[@]};
do
    cmd="$cmd_prefix $script --some_parameter_name ${param}"
    $cmd        
done
