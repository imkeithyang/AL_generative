#!/bin/bash
echo -n "Yaml File: ";
read -r file;
echo -n "cuda: ";
read -r i;
screen_name=${file////-}
echo "$screen_name";
screen -S $screen_name -dm bash -c "python main_attflow_cond_stim.py -f $file -d cuda:$i; sleep 500";