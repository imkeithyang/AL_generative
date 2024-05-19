#!/bin/bash
echo -n "cuda: ";
read -r i;

#seed range from 0 to 9
for seed in {40..49}; do
    echo "$seed";

    #create screens, and record output of each screen to a separate text file
    screen -S $seed -dm bash -c "python main_attflow_cond_stim_seedwise.py -p /hpc/home/pc266/data/ALdata/070921_cleaned.csv -d cuda:$i -prefix /hpc/group/tarokhlab/pc266 -seed $seed | tee text_outputs/$seed.txt; sleep 500 ";

done
