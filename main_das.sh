#!/bin/bash

# Activate upper bash command to set conda environment.
#eval "$(/home/lvdwater/miniconda3/bin/conda shell.bash hook)"

# Specify the path to the base directory of the hyper-heuristic DSE here.
BASE_PATH="/home/mherget/hyper-heuristic-dse-2.0"
# Specify the path to the heuristic run configuration file here
CONFIG_PATH_HEUR_RUN="${BASE_PATH}/config/config_heuristic_run.json"
# Specify the path to the coordinator configuration file here
CONFIG_PATH_COORDINATOR="${BASE_PATH}/config/config_coordinator.json"

# Define PYTHONPATH to Herman's framework.
export PYTHONPATH="${PYTHONPATH}:${BASE_PATH}/src_main/external/simulation_model"

# Define OMNET++ shared library to link loader.
OMNET_BASE="/home/mherget/omnet/omnetpp-6.0.1"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${OMNET_BASE}/lib/

# conda config --set auto_activate_base false
#cd ${OMNET_BASE}
#source setenv
#cd ../inet4.5
#source setenv

#cd ~/hyper-heuristic-dse-2.0
python3 main.py --base_path "$BASE_PATH" --heur_run_config "$CONFIG_PATH_HEUR_RUN" --coordinator_config "$CONFIG_PATH_COORDINATOR"
