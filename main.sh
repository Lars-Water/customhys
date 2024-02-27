#!/bin/bash

# Specify the path to the heuristic run configuration file here
CONFIG_PATH_HEUR_RUN="/home/lvdwater/hyper-heuristic-dse-2.0/config/config_heuristic_run.json"
# Specify the path to the coordinator configuration file here
CONFIG_PATH_COORDINATOR="/home/lvdwater/hyper-heuristic-dse-2.0/config/config_coordinator.json"

# Activate upper bash command to set conda environment.
# eval "$(/home/lvdwater/miniconda3/bin/conda shell.bash hook)"
# conda config --set auto_activate_base false
cd /var/scratch/lvdwater/omnetpp-6.0.1/
source setenv
cd ../inet4.5
source setenv
cd ~/hyper-heuristic-dse-2.0
python3 src/main.py --heur_run_config "$CONFIG_PATH_HEUR_RUN" --coordinator_config "$CONFIG_PATH_COORDINATOR"
