CUR=$(pwd)
cd /home/larry/hyper-heuristic-dse-2.0/omnetpp-6.0.1
source setenv
cd /home/larry/hyper-heuristic-dse-2.0/omnetpp-6.0.1/inet4.5
source setenv
cd $CUR
python3 experiments.py --platform local --sims_path "/home/larry/hyper-heuristic-dse-2.0/src_main/external/simulation_model/sims" --dummy_path "/home/larry/hyper-heuristic-dse-2.0/src_main/external/simulation_model/sims" --inet_path "/home/larry/hyper-heuristic-dse-2.0/omnetpp-6.0.1/inet4.5" --experiments_path /home/larry/hyper-heuristic-dse-2.0/src_main/external/simulation_model/experiments
