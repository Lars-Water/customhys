#!/bin/bash

# Activate upper bash command to set conda environment.
eval "$(/home/lvdwater/miniconda3/bin/conda shell.bash hook)"

# Specify the path to the base directory of the hyper-heuristic DSE here.
BASE_PATH="/home/lvdwater/hyper-heuristic-dse-2.0"

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
cd /home/lvdwater/hyper-heuristic-dse-2.0/src_main/external/simulation_model

python3 workflow.py --platform local --sims_path "/var/scratch/lvdwater/sims" --dummy_path "/home/lvdwater/hyper-heuristic-dse-2.0/src_main/external/simulation_model/sims" --inet_path "/var/scratch/lvdwater/inet4.5" --workflowConfigFile  /home/lvdwater/hyper-heuristic-dse-2.0/src_main/external/simulation_model/workflow/config/workflow_das.json --workflowLogsFolder /var/scratch/lvdwater/workflow/logs --workflowResultsFolder /var/scratch/lvdwater/tuning_workflow --workflowRuntimeFolder /var/scratch/lvdwater/workflow/runtime