cd /var/scratch/lvdwater/omnetpp-6.0.1/
source setenv
cd ../inet4.5/
source setenv
cd /home/lvdwater/hyper-heuristic-dse-2.0/src_main/external/simulation_model/
python3 experiments.py --platform DAS --sims_path "/var/scratch/lvdwater/sims" --dummy_path "/home/lvdwater/hyper-heuristic-dse-2.0/src_main/external/simulation_model/sims" --inet_path "/var/scratch/lvdwater/inet4.5/" --experiments_path /var/scratch/lvdwater/experiments