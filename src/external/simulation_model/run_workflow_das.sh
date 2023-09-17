cd ../omnet/omnetpp-6.0.1
source setenv
cd ../inet4.5
source setenv
cd ../../MAINFOLDER
python3 workflow.py --platform DAS --sims_path "/var/scratch/USER/sims" --dummy_path "/home/USER/Scalable-Simulation-DCPS/sims" --inet_path "/home/USER/omnet/inet4.5" --workflowConfigFile  /home/USER/Scalable-Simulation-DCPS/workflow/config/workflow_das.json --workflowLogsFolder /var/scratch/USER/workflow/logs --workflowResultsFolder /var/scratch/USER/workflow/results --workflowRuntimeFolder /var/scratch/USER/workflow/runtime