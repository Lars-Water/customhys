#!/bin/bash

# Activate upper bash command to set conda environment.
eval "$(/home/mherget/miniconda3/bin/conda shell.bash hook)"

# Specify the path to the base directory of the hyper-heuristic DSE here.
BASE_PATH="/home/mherget/scratch/hyper-heuristic-dse-2.0"

# Specify the path to the coordinator configuration file here
CONFIG_PATH_COORDINATOR="${BASE_PATH}/config/coordinator/config_coordinator_das_2_herget_exp1.json"

# Define PYTHONPATH to Herman's framework.
export PYTHONPATH="${PYTHONPATH}:${BASE_PATH}/src_main/external"
export PYTHONPATH="${PYTHONPATH}:${BASE_PATH}/src_main/external/simulation_model"
export PYTHONPATH="${PYTHONPATH}:${BASE_PATH}/src_main/external/customhys_local"
export PYTHONPATH="${PYTHONPATH}:${BASE_PATH}"

# Define OMNET++ shared library to link loader.
OMNET_BASE="/home/herget/omnet/omnetpp-6.0.1"
export PATH=$PATH:${OMNET_BASE}/bin/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${OMNET_BASE}/lib/

cd ${OMNET_BASE}
source setenv
cd /home/herget/omnet/omnet-projects/inet4.5
source setenv
cd ${BASE_PATH}
eval "ulimit -n 262000"
ulimit -n 262000

# EXPERIMENT PIPELINES

python3 main.py \
 --experiment 1 \
 --nr_sw  50 \
 --base_path "$BASE_PATH" \
 --coordinator_config "$CONFIG_PATH_COORDINATOR"