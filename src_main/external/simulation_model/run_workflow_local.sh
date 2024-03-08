CUR=$(pwd)
cd /home/larry/hyper-heuristic-dse-2.0/omnetpp-6.0.1
source setenv
cd /home/larry/hyper-heuristic-dse-2.0/omnetpp-6.0.1/inet4.5
source setenv
cd $CUR
python3 workflow.py --platform local --sims_path "/home/larry/hyper-heuristic-dse-2.0/src_main/external/simulation_model/sims" --dummy_path "/home/larry/hyper-heuristic-dse-2.0/src_main/external/simulation_model/sims" --inet_path "/home/larry/hyper-heuristic-dse-2.0/omnetpp-6.0.1/inet4.5" --workflowConfigFile  /home/larry/hyper-heuristic-dse-2.0/src_main/external/simulation_model/workflow/config/workflow_local.json --workflowLogsFolder /home/larry/hyper-heuristic-dse-2.0/src_main/external/simulation_model/workflow/logs --workflowResultsFolder  /home/larry/hyper-heuristic-dse-2.0/data/raw/tuning_workflow --workflowRuntimeFolder  /home/larry/hyper-heuristic-dse-2.0/src_main/external/simulation_model/workflow/runtime