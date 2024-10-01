#!/bin/bash

# Specify the path to the base directory of the hyper-heuristic DSE here.
BASE_PATH="/home/larry/hyper-heuristic-dse-2.0"
# Specify the path to the heuristic run configuration file here
CONFIG_PATH_HEUR_RUN="${BASE_PATH}/config/config_heuristic_run.json"
# Specify the path to the coordinator configuration file here
CONFIG_PATH_COORDINATOR="${BASE_PATH}/config/coordinator/config_coordinator_local.json"

export PYTHONPATH="${PYTHONPATH}:${BASE_PATH}/src_main/external/simulation_model"

cd omnetpp-6.0.1/
source setenv
cd inet4.5/
source setenv

cd ../../
# # DESIGN SPACE PLOTTING.
# python3 main.py --nr_sw 6 --base_path "$BASE_PATH" --coordinator_config "$CONFIG_PATH_COORDINATOR" --design_space_plot
# # PARAMETER TUNING.
# python3 main.py --nr_sw 10 --base_path "$BASE_PATH" --coordinator_config "$CONFIG_PATH_COORDINATOR" --param_tune
# # HEURISTIC RUN
# python3 main.py --nr_sw 6 --base_path "$BASE_PATH" --coordinator_config "$CONFIG_PATH_COORDINATOR" --heur_run_config "$CONFIG_PATH_HEUR_RUN"

# Specify the path to the hyperheuristic run configuration file here.
CONFIG_PATH_HYPHEUR_RUN="${BASE_PATH}/config/hh/config_hh_run.json"
# # Specify the path to the hyperheuristic run configuration file here.
# CONFIG_PATH_HYPHEUR_RUN="${BASE_PATH}/config/hh/config_hh_run_test_mh_collection.json"

# # HYPER-HEURISTIC RUN
# python3 main.py --nr_sw 10 --base_path "$BASE_PATH" --coordinator_config "$CONFIG_PATH_COORDINATOR" --heur_run_config "$CONFIG_PATH_HYPHEUR_RUN"

# Metaheuristics to visualize
METAHEURISTICS="gravitational pso ga"

# HH Run directories to visualize in experiment 1.
HH_RUN_DIRS_EXP_1="$PWD/data_files/raw/INET-LANS_experiment_1_genetic_crossover_1726576957"

# HH Run directories to visualize in experiment 2.
HH_RUN_DIRS_EXP_2="$PWD/data_files/raw/INET-LANS_experiment_2_50_backbones_hh_run_with_followed_up_positions_10_iterations_20_steps_1727434496 $PWD/data_files/raw/INET-LANS_experiment_2_200_backbones_hh_run_with_followed_up_positions_10_iterations_20_steps_1727439213 $PWD/data_files/raw/INET-LANS_experiment_2_300_backbones_hh_run_with_followed_up_positions_10_iterations_20_steps_1727525178"

# Run the Python script with metaheuristics
python3 main.py \
  --visualize 2 \
  --metaheuristics $METAHEURISTICS \
  --hh_run_dirs_exp_1 $HH_RUN_DIRS_EXP_1 \
  --hh_run_dirs_exp_2 $HH_RUN_DIRS_EXP_2 \
  --base_path "$BASE_PATH" \
  --coordinator_config "$CONFIG_PATH_COORDINATOR"

# # Run the Python script with metaheuristics
# python3 main.py \
#   --experiment all \
#   --visualize all \
#   --metaheuristics $METAHEURISTICS \
#   --base_path "$BASE_PATH" \
#   --coordinator_config "$CONFIG_PATH_COORDINATOR"
