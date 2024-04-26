#!/bin/bash
echo -n "Experiment: ";
read -r exp;
echo -n "cuda: ";
read -r i;
for file in $exp/*.yaml; do 
    screen_name=${file////-}
    echo "$screen_name";
    screen -S $screen_name -dm bash -c "python main_attflow_cond_stim.py -f $file -d cuda:$i -seed 42 -shuffle True; sleep 500";
    i=$((i+1))
    if [[ $i == 4 ]] 
    then
        i=0
    fi
done