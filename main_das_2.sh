#!/bin/bash

# Activate upper bash command to set conda environment.
eval "$(/home/lvdwater/miniconda3/bin/conda shell.bash hook)"


# Specify the path to the base directory of the hyper-heuristic DSE here.
BASE_PATH="/home/lvdwater/hyper-heuristic-dse-2.0"

# Specify the path to the coordinator configuration file here
CONFIG_PATH_COORDINATOR="${BASE_PATH}/config/coordinator/config_coordinator_das_2.json"

# Define PYTHONPATH to Herman's framework.
export PYTHONPATH="${PYTHONPATH}:${BASE_PATH}/src_main/external/simulation_model"

# Define OMNET++ shared library to link loader.
OMNET_BASE="/var/scratch/lvdwater/omnetpp-6.0.1"
export PATH=$PATH:/var/scratch/lvdwater/omnet/omnetpp-6.0.1/bin/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${OMNET_BASE}/lib/

cd ${OMNET_BASE}
source setenv
cd ../inet4.5
source setenv
cd ${BASE_PATH}
eval "ulimit -n 262000"
ulimit -n 262000


python3 main.py \
 --experiment 1 \
 --nr_sw  50 \
 --base_path "$BASE_PATH" \
 --coordinator_config "$CONFIG_PATH_COORDINATOR"
