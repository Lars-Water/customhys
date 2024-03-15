#!/bin/bash

# Specify the path to the base directory of the hyper-heuristic DSE here.
BASE_PATH="/home/larry/hyper-heuristic-dse-2.0"
# Specify the path to the heuristic run configuration file here
CONFIG_PATH_HEUR_RUN="${BASE_PATH}/config/config_heuristic_run.json"
# Specify the path to the coordinator configuration file here
CONFIG_PATH_COORDINATOR="${BASE_PATH}/config/config_coordinator_local.json"

export PYTHONPATH="${PYTHONPATH}:${BASE_PATH}/src_main/external/simulation_model"

cd omnetpp-6.0.1/
source setenv
cd inet4.5/
source setenv

cd ../../
# # DESIGN SPACE PLOTTING.
# python3 main.py --nr_sw 6 --base_path "$BASE_PATH" --coordinator_config "$CONFIG_PATH_COORDINATOR" --design_space_plot
# PARAMETER TUNING.
python3 main.py --nr_sw 10 --base_path "$BASE_PATH" --coordinator_config "$CONFIG_PATH_COORDINATOR" --param_tune
# # HEURISTIC RUN
# python3 main.py --base_path "$BASE_PATH" --coordinator_config "$CONFIG_PATH_COORDINATOR" --heur_run_config "$CONFIG_PATH_HEUR_RUN"
