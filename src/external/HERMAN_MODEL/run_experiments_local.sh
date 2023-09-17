# LOCAL WSL UBUNTU
# CUR=$(pwd)
# cd /home/lars-water/omnetpp-6.0.1
# source setenv
# cd omnet-projects/inet4.5
# source setenv
# cd $CUR
# python3 experiments.py --platform local --sims_path "/mnt/c/Users/larry/Desktop/SE_thesis/hh-dse-2.0/src/external/Scalable-Simulation-DCPS-main/sims" --dummy_path "/mnt/c/Users/larry/Desktop/SE_thesis/hh-dse-2.0/src/external/Scalable-Simulation-DCPS-main/sims" --inet_path "/home/lars-water/omnetpp-6.0.1/omnet-projects/inet4.5" --experiments_path /mnt/c/Users/larry/Desktop/SE_thesis/hh-dse-2.0/src/external/Scalable-Simulation-DCPS-main/experiments
# GITHUB CODESPACES
CUR=$(pwd)
cd /workspaces/hyper-heuristic-dse-2.0/omnetpp-6.0.1
source setenv
cd /workspaces/hyper-heuristic-dse-2.0/omnetpp-6.0.1/inet4.5
source setenv
cd $CUR
python3 experiments.py --platform local --sims_path "/workspaces/hyper-heuristic-dse-2.0/src/external/HERMAN_MODEL/sims" --dummy_path "/workspaces/hyper-heuristic-dse-2.0/src/external/HERMAN_MODEL/sims" --inet_path "/workspaces/hyper-heuristic-dse-2.0/omnetpp-6.0.1/inet4.5" --experiments_path /workspaces/hyper-heuristic-dse-2.0/src/external/HERMAN_MODEL/experiments
# GITHUB CODESPACES experiment_seq
# CUR=$(pwd)
# cd /workspaces/hyper-heuristic-dse-2.0/omnetpp-6.0.1
# source setenv
# cd /workspaces/hyper-heuristic-dse-2.0/omnetpp-6.0.1/inet4.5
# source setenv
# cd $CUR
# python experiment_seq.py \
#   --data_path=/workspaces/hyper-heuristic-dse-2.0/src/external/HERMAN_MODEL/experiments/ \
#   --inet_path=/workspaces/omnetpp-6.0.1/inet4.5/ \
#   --sims_path=/workspaces/hyper-heuristic-dse-2.0/src/external/HERMAN_MODEL/sims/ \
#   --dummy_path=/workspaces/hyper-heuristic-dse-2.0/src/external/HERMAN_MODEL/sims/ \
#   --platform=local
