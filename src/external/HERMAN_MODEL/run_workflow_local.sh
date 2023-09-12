# LOCAL
# CUR=$(pwd)
# cd /home/lars-water/omnetpp-6.0.1
# source setenv
# cd omnet-projects/inet4.5
# source setenv
# cd $CUR
# python3 workflow.py --platform local --sims_path "/mnt/c/Users/larry/Desktop/SE_thesis/hh-dse-2.0/src/external/Scalable-Simulation-DCPS-main/sims" --dummy_path "/mnt/c/Users/larry/Desktop/SE_thesis/hh-dse-2.0/src/external/Scalable-Simulation-DCPS-main/sims" --inet_path "/home/lars-water/omnetpp-6.0.1/omnet-projects/inet4.5" --workflowConfigFile  /mnt/c/Users/larry/Desktop/SE_thesis/hh-dse-2.0/src/external/Scalable-Simulation-DCPS-main/workflow/config/workflow_local.json --workflowLogsFolder /mnt/c/Users/larry/Desktop/SE_thesis/hh-dse-2.0/src/external/Scalable-Simulation-DCPS-main/workflow/logs --workflowResultsFolder  /mnt/c/Users/larry/Desktop/SE_thesis/hh-dse-2.0/src/external/Scalable-Simulation-DCPS-main/workflow/results --workflowRuntimeFolder  /mnt/c/Users/larry/Desktop/SE_thesis/hh-dse-2.0/src/external/Scalable-Simulation-DCPS-main/workflow/runtime
# GOOGLE CLOUD CONSOLE - NOT WORKING
# CUR=$(pwd)
# cd /home/larryvanderwater/omnetpp-6.0.1
# source setenv
# cd home/larryvanderwater/omnetpp-6.0.1/inet4.5
# source setenv
# cd $CUR
# python3 workflow.py --platform local --sims_path "/home/larryvanderwater/hyper-heuristic-dse-2.0/src/external/HERMAN_MODEL/sims" --dummy_path "/home/larryvanderwater/hyper-heuristic-dse-2.0/src/external/HERMAN_MODEL/sims" --inet_path "home/larryvanderwater/omnetpp-6.0.1/inet4.5" --workflowConfigFile  /home/larryvanderwater/hyper-heuristic-dse-2.0/src/external/HERMAN_MODEL/workflow/config/workflow_local.json --workflowLogsFolder /home/larryvanderwater/hyper-heuristic-dse-2.0/src/external/HERMAN_MODEL/workflow/logs --workflowResultsFolder  /home/larryvanderwater/hyper-heuristic-dse-2.0/src/external/HERMAN_MODEL/workflow/results --workflowRuntimeFolder  /home/larryvanderwater/hyper-heuristic-dse-2.0/src/external/HERMAN_MODEL/workflow/runtime
# GITHUB CODESPACES
CUR=$(pwd)
cd /workspaces/hyper-heuristic-dse-2.0/omnetpp-6.0.1
source setenv
cd /workspaces/hyper-heuristic-dse-2.0/omnetpp-6.0.1/inet4.5
source setenv
cd $CUR
python3 workflow.py --platform local --sims_path "/workspaces/hyper-heuristic-dse-2.0/src/external/HERMAN_MODEL/sims" --dummy_path "/workspaces/hyper-heuristic-dse-2.0/src/external/HERMAN_MODEL/sims" --inet_path "home/larryvanderwater/omnetpp-6.0.1/inet4.5" --workflowConfigFile  /workspaces/hyper-heuristic-dse-2.0/src/external/HERMAN_MODEL/workflow/config/workflow_local.json --workflowLogsFolder /workspaces/hyper-heuristic-dse-2.0/src/external/HERMAN_MODEL/workflow/logs --workflowResultsFolder  /workspaces/hyper-heuristic-dse-2.0/src/external/HERMAN_MODEL/workflow/results --workflowRuntimeFolder  /workspaces/hyper-heuristic-dse-2.0/src/external/HERMAN_MODEL/workflow/runtime