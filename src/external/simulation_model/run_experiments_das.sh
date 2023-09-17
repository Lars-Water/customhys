cd /var/scratch/USER/omnetpp-6.0.1/
source setenv
cd ~/omnet/inet4.5/
source setenv
cd ~/USER
python3 experiments.py --platform DAS --sims_path "/var/scratch/USER/sims" --dummy_path "/home/USER/sims" --inet_path "/USER/omnet/inet4.5" --experiments_path /var/scratch/USER/experiments