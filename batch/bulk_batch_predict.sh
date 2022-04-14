#!/bin/bash -x
# Submits batch predict with various parameter configurations to the cluster
#
# Usage:
#   src/models/online_expert/bulk_batch_predict.sh std_val flip_flop
#   src/models/online_expert/bulk_batch_predict.sh std_test flip_flop
#   src/models/online_expert/bulk_batch_predict.sh std_ens flip_flop
#   src/models/online_expert/bulk_batch_predict.sh std_future flip_flop
#
# Args:
#   target_dates: one of the standard target date sets (std_val, std_test, or 
#     std_ens) or a comma-separated string of dates in YYYYMMDD format
#     (e.g., "20191112,20191126"); std_val by default

target_dates=${1:-std_ens}
alg=${2:-adahedgefo}
exp_name=${3:-"None"}
#cmd_prefix="src/batch/batch_python.sh --memory 3 --cores 1 --hours 0 --minutes 7"
cmd_prefix="python"
#script="src/models/online_expert/batch_predict.py"
script="src/models/online_expert/online_expert_local.py"

# List of parameter settings to try
if [ ${alg} = "adahedgefo" ]; then
    reg_list=("forward_drift" "plusplus")
elif [ ${alg} = "dadahedge" ]; then
    reg_list=("adaptive" "upper_bound")
else
    reg_list=("None")
fi
#hint_list=("None" "prev_g" "mean_g" "adaptive")
hint_list=("adaptive")
#hint_list=("None" "current_w" "uniform" "prev_y" "trend_y" "mean_y" "prev_g" "mean_g" "doy" "adaptive")
#hint_list=("trend_y")
#hint_list=("None" "current_w" "uniform" "prev_y" "trend_y" "mean_y" "prev_g" "mean_g" "doy")
#hint_list=("adaptive")
#hint_list=("None")
model_list=(
  "tuned_catboost,tuned_cfsv2,tuned_doy,llr,multillr,tuned_salient_fri"
)
for horizon in 34w 56w
do
    task_str="contest_precip $horizon -t $target_dates"
    if [ ${exp_name} = "UNIFORM" ]; then
        delay_list=(1000)
        hint_list=("None")
    elif [ ${horizon} = "34w" ]; then
        #delay_list=(0 2)
        delay_list=(0)
    else
        #delay_list=(0 3)
        delay_list=(0)
    fi

    for model in ${model_list[@]};
    do
        for quarter in ${quarter_list[@]};
        do
            for hint in ${hint_list[@]};
            do
                for delay in ${delay_list[@]};
                do
                    #if [ ${delay} = 1000 && ${hint} != "None" ]; then
                    #  continue
                    #fi
                    for reg in ${reg_list[@]};
                    do
                        if [ ${hint} = "adaptive" ]; then
                            #meta_list=("False" "True")
                            meta_list=("False")
                        else
                            meta_list=("False")
                        fi

                        for meta in ${meta_list[@]};
                        do
                            cmd="$cmd_prefix $script $task_str --expert_models ${model} --alg $alg --quarters ${quarter} --hint ${hint} --exp_name ${exp_name} --delay ${delay} --reg ${reg} --meta ${meta}"
                            $cmd        
                        done
                    done
                done
            done    
        done
    done
done
