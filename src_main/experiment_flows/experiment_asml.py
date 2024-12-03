import os
import time
import threading
from pathlib import Path

from src_main.external.customhys.customhys import hyperheuristic as hh

from src_main.data import collect_data_INET
from src_main.tools import component_config
from src_main.tools import coordinator_asml
from src_main.tools.config_reader import Config

save_runs = []
experiment_config = None
max_cost = None
def run_experiment(experiment_config, coordinator_params):
    '''
        Experimental run of tuning the parameters of any provided search operators.
    '''
    search_operators_spaces = experiment_config.tryGet('search_operators', 'spaces')
    pass_finalised_positions = experiment_config.tryGet('pass_finalised_positions')
    hh_parameters = experiment_config.tryGet('hh_parameters')
    num_replicas = hh_parameters["num_replicas"] if hh_parameters["num_replicas"] > 0 else 1 

    
    # Create the HeuristicSimulationCoordinator.
    base_path, coordinator_config_file_path, nr_of_agents, run_name = coordinator_params
    heur_sim_coordinator = coordinator_asml.HeuristicSimulationCoordinatorASML(base_path, coordinator_config_file_path, nr_of_agents, 
        run_name=run_name,
        nr_of_design_queues = len(search_operators_spaces.keys())*num_replicas,
        experiment_config=experiment_config
    ) # noqa 501
    heur_sim_coordinator.set_run_name("ASML_"+str(heur_sim_coordinator.time_stamp))

    heur_sim_coordinator.manual_normalization()
    min_wfpm_runtime, max_wfpm_runtime, min_cost, max_cost = heur_sim_coordinator.get_boundaries()
    # min_wfpm_runtime = 2
    # max_wfpm_runtime = 2 
    # min_cost = 2
    # max_cost = 2

    # Reset the execution times before starting the heuristic run.
    heur_sim_coordinator.coordinator_and_simulation_execution_time = 0
    heur_sim_coordinator.simulation_execution_time = 0
    
    timestamp = int(time.time())

    conf = Config(coordinator_config_file_path, name = f"run_experiment_asml")
    log_path = conf.tryGet("output_paths", "log_files")
    if log_path is None:
        log_path = os.path.join(base_path, "data/logs/")

    coordinator_log_path = Path(os.path.join(log_path, "exp_asml"))
    conf.createLogger(coordinator_log_path, f"run_experiment_asml")

    fns_hh = []
    for search_operator_space_name in search_operators_spaces.keys():
        space = search_operators_spaces[search_operator_space_name]
        search_operator_space_path = space["path"]
        fns_hh.append(threading.Thread(target=run_search_operator_space_path, args=(
    experiment_config, search_operator_space_path, search_operator_space_name, max_wfpm_runtime, min_wfpm_runtime, min_cost, max_cost, heur_sim_coordinator, coordinator_params, timestamp, conf)))


    # print(fns_hh)
    proc = []
    for p in fns_hh:
        p.start()
        proc.append(p)
    # print("##### procs")
    # print(proc)
    for p in proc:
        p.join()

    print(save_runs)
    print("Finished Exp 1")

def run_search_operator_space_path(experiment_config, search_operator_space_path, search_operator_space_name, max_wfpm_runtime, min_wfpm_runtime, min_cost, max_cost, heur_sim_coordinator, coordinator_params, timestamp, conf):
    base_path, coordinator_config_file_path, nr_of_agents, run_name = coordinator_params

    pass_finalised_positions = experiment_config.tryGet('pass_finalised_positions')

    # Get the current Unix timestamp
    experiment_name = f"experiment_asml_{search_operator_space_name}_{str(timestamp)}"

    print("##### ("+search_operator_space_name+") determine_heuristic_space")

    heuristic_space = component_config.determine_heuristic_space(search_operator_space_path)

    # Configure Hyperheurstic Object.
    hh_parameters = experiment_config.tryGet('hh_parameters')
    nr_of_iterations = experiment_config.tryGet('hh_parameters', 'num_iterations')
    nr_of_steps = experiment_config.tryGet('hh_parameters', 'num_steps')
    file_label = f"ASML-Faezeh_{experiment_name}_{nr_of_iterations}_iterations_{nr_of_steps}_steps_{str(timestamp)}"

    num_replicas = hh_parameters["num_replicas"] if hh_parameters["num_replicas"] > 0 else 1 
    simulation_model_template_path = conf.tryGet("simulation_model", "simulation_model_paths", "simulation_model_template_path")
    template_file_path = os.path.join(simulation_model_template_path, "platform.xml")
    agents_fitness_values_path = conf.tryGet("output_paths", "agents_fitness_values_path")
    
    heur_sim_coordinator.search_operator_spaces.create_problems(search_operator_space_name, num_replicas, component_config.create_problem_instanceASML, template_file_path, max_wfpm_runtime, min_wfpm_runtime, min_cost, max_cost, heur_sim_coordinator.simulation_run, agents_fitness_values_path)

    
    ret = heur_sim_coordinator.hh(heuristic_space, search_operator_space_name, hh_parameters, file_label, pass_finalised_positions, timestamp, template_file_path, experiment_name)
   

    save_runs.append(ret)
