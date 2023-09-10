CUR=$(pwd)
cd /home/lars-water/omnetpp-6.0.1
source setenv
cd omnet-projects/inet4.5
source setenv
cd $CUR
python3 experiments.py --platform local --sims_path "/mnt/c/Users/larry/Desktop/SE_thesis/hh-dse-2.0/src/external/Scalable-Simulation-DCPS-main/sims" --dummy_path "/mnt/c/Users/larry/Desktop/SE_thesis/hh-dse-2.0/src/external/Scalable-Simulation-DCPS-main/sims" --inet_path "/home/lars-water/omnetpp-6.0.1/omnet-projects/inet4.5" --experiments_path /mnt/c/Users/larry/Desktop/SE_thesis/hh-dse-2.0/src/external/Scalable-Simulation-DCPS-main/experiments
