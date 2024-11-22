#!/bin/bash

# Activate upper bash command to set conda environment.
eval "$(/home/mherget/miniconda3/bin/conda shell.bash hook)"

# Specify the path to the base directory of the hyper-heuristic DSE here.
BASE_PATH="/home/herget/UvA-git/hyper-heuristic-dse-2.0"

# Specify the path to the coordinator configuration file here
CONFIG_PATH_COORDINATOR="${BASE_PATH}/config/coordinator/config_coordinator_das_2_herget.json"

# Define PYTHONPATH to Herman's framework.
export PYTHONPATH="${PYTHONPATH}:${BASE_PATH}/src_main/external/simulation_model"

# Define OMNET++ shared library to link loader.
OMNET_BASE="/var/scratch/mherget/omnet/omnetpp-6.0.1"
export PATH=$PATH:/var/scratch/mherget/omnet/omnetpp-6.0.1/bin/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${OMNET_BASE}/lib/

cd ${OMNET_BASE}
source setenv
cd ../inet4.5
source setenv
cd ${BASE_PATH}
eval "ulimit -n 262000"
ulimit -n 262000


# EXPERIMENT PIPELINES

# Experiment 1.
CONFIG_PATH_EXPERIMENT_1="${BASE_PATH}/config/experiment_1/experiment_2.json"

python3 main_vis.py \
 --visualize "asml" \
 --hh_run_dirs_exp_1 "/home/herget/UvA-git/hyper-heuristic-dse-2.0/data_files/raw/ASML-Faezeh_experiment_asml_genetic_crossover_1732116689_20_iterations_10_steps_1732116689"\
    "/home/herget/UvA-git/hyper-heuristic-dse-2.0/data_files/raw/ASML-Faezeh_experiment_asml_genetic_mutation_1732116689_20_iterations_10_steps_1732116689"\
    "/home/herget/UvA-git/hyper-heuristic-dse-2.0/data_files/raw/ASML-Faezeh_experiment_asml_gravitational_1732116689_20_iterations_10_steps_1732116689"\
    "/home/herget/UvA-git/hyper-heuristic-dse-2.0/data_files/raw/ASML-Faezeh_experiment_asml_swarm_dynamic_1732116689_20_iterations_10_steps_1732116689"\
 --result_dir "/home/herget/UvA-git/hyper-heuristic-dse-2.0/data_files/graphs" \
  --base_path "$BASE_PATH" \
 --coordinator_config "$CONFIG_PATH_COORDINATOR"
 
# python3 main_vis.py \
#  --visualize "asml" \
#  --hh_run_dirs_exp_1  \
#  --result_dir "/home/herget/UvA-git/hyper-heuristic-dse-2.0/data_files/graphs" \
#   --base_path "$BASE_PATH" \
#  --coordinator_config "$CONFIG_PATH_COORDINATOR"

# python3 main_vis.py \
#  --visualize "asml" \
#  --hh_run_dirs_exp_1  \
#  --result_dir "/home/herget/UvA-git/hyper-heuristic-dse-2.0/data_files/graphs" \
#   --base_path "$BASE_PATH" \
#  --coordinator_config "$CONFIG_PATH_COORDINATOR"

# python3 main_vis.py \
#  --visualize "asml" \
#  --hh_run_dirs_exp_1  \
#  --result_dir "/home/herget/UvA-git/hyper-heuristic-dse-2.0/data_files/graphs" \
#   --base_path "$BASE_PATH" \
#  --coordinator_config "$CONFIG_PATH_COORDINATOR"
