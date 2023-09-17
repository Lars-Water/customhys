CUR=$(pwd)
cd /workspaces/hyper-heuristic-dse-2.0/omnetpp-6.0.1
source setenv
cd /workspaces/hyper-heuristic-dse-2.0/omnetpp-6.0.1/inet4.5
source setenv
cd $CUR
python3 experiments.py --platform local --sims_path "/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/sims" --dummy_path "/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/sims" --inet_path "/workspaces/hyper-heuristic-dse-2.0/omnetpp-6.0.1/inet4.5" --experiments_path /workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/experiments
