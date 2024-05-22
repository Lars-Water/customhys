#!/bin/bash

# Activate upper bash command to set conda environment.
eval "$(/home/lvdwater/miniconda3/bin/conda shell.bash hook)"

# Specify the path to the base directory of the hyper-heuristic DSE here.
BASE_PATH="/home/lvdwater/hyper-heuristic-dse-2.0"
# Specify the path to the coordinator configuration file here
CONFIG_PATH_COORDINATOR="${BASE_PATH}/config/config_coordinator_das.json"

# Define PYTHONPATH to Herman's framework.
export PYTHONPATH="${PYTHONPATH}:${BASE_PATH}/src_main/external/simulation_model"

# Define OMNET++ shared library to link loader.
OMNET_BASE="/var/scratch/lvdwater/omnetpp-6.0.1"
export PATH=$PATH:/var/scratch/lvdwater/omnetpp-6.0.1/bin/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${OMNET_BASE}/lib/

cd ${OMNET_BASE}
source setenv
cd ../inet4.5
source setenv
cd ~/hyper-heuristic-dse-2.0
eval "ulimit -n 50000"
ulimit -n 50000

# CONFIG_PATH_HEUR_RUN="${BASE_PATH}/config/config_heuristic_run.json"
# python3 main.py --nr_sw 50 --base_path "$BASE_PATH" --heur_run_config "$CONFIG_PATH_HEUR_RUN" --coordinator_config "$CONFIG_PATH_COORDINATOR"

CONFIG_PATH_HEUR_RUN_PSO="${BASE_PATH}/config/config_heuristic_run_pso.json"
python3 main.py --nr_sw 50 --base_path "$BASE_PATH" --heur_run_config "$CONFIG_PATH_HEUR_RUN_PSO" --coordinator_config "$CONFIG_PATH_COORDINATOR"

# CONFIG_PATH_HEUR_RUN_GA="${BASE_PATH}/config/config_heuristic_run_ga.json"
# python3 main.py --nr_sw 50 --base_path "$BASE_PATH" --heur_run_config "$CONFIG_PATH_HEUR_RUN_GA" --coordinator_config "$CONFIG_PATH_COORDINATOR"
