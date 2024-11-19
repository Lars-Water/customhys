import os
import time
import threading
from pathlib import Path

from src_main.external.customhys.customhys import hyperheuristic as hh

from src_main.data import collect_data
from src_main.tools import component_config
from src_main.tools import coordinator_asml
from src.utils.config_reader import Config

save_runs = []
experiment_config = None
max_cost = None
def run_experiment(experiment_config, coordinator_params):
    '''
        Experimental run of tuning the parameters of any provided search operators.
    '''
    search_operator_space_paths = experiment_config.tryGet('search_operator_space_paths')
    search_operator_space_names = experiment_config.tryGet('search_operator_space_names')
    pass_finalised_positions = experiment_config.tryGet('pass_finalised_positions')
    hh_parameters = experiment_config.tryGet('hh_parameters')
    num_replicas = hh_parameters["num_replicas"] if hh_parameters["num_replicas"] > 0 else 1 

    
    # Create the HeuristicSimulationCoordinator.
    base_path, coordinator_config_file_path, nr_of_agents, run_name = coordinator_params
    heur_sim_coordinator = coordinator_asml.HeuristicSimulationCoordinatorASML(base_path, coordinator_config_file_path, nr_of_agents, run_name, len(search_operator_space_names)*num_replicas) # noqa 501
    # Create problem instance.
    heur_sim_coordinator.manual_normalization()
    min_wfpm_runtime, max_wfpm_runtime, min_cost, max_cost = heur_sim_coordinator.get_boundaries()

    # Reset the execution times before starting the heuristic run.
    heur_sim_coordinator.coordinator_and_simulation_execution_time = 0
    heur_sim_coordinator.simulation_execution_time = 0
    

    fns_hh = []
    for search_operator_space_path, search_operator_space_name in zip(search_operator_space_paths, search_operator_space_names):
        fns_hh.append(threading.Thread(target=run_search_operator_space_path, args=(
    experiment_config, search_operator_space_path, search_operator_space_name, max_wfpm_runtime, min_wfpm_runtime, min_cost, max_cost, heur_sim_coordinator, coordinator_params)))


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

def run_search_operator_space_path(experiment_config, search_operator_space_path, search_operator_space_name, max_wfpm_runtime, min_wfpm_runtime, min_cost, max_cost, heur_sim_coordinator, coordinator_params):
    base_path, coordinator_config_file_path, nr_of_agents, run_name = coordinator_params
    coordinator_log_path = os.path.join(base_path, "data/logs/exp_asml")              
    os.makedirs(coordinator_log_path, exist_ok=True)

    conf = Config(coordinator_config_file_path, Path(coordinator_log_path), f"run_search_operator_space_path_{search_operator_space_name}")

    pass_finalised_positions = experiment_config.tryGet('pass_finalised_positions')
        # Open a file in write mode ('w') and write the string
    with open("output.txt", "a") as file:
        file.write(f"Running {search_operator_space_path} - {search_operator_space_name}")

    # Get the current Unix timestamp
    timestamp = int(time.time())
    experiment_name = f"experiment_asml_{search_operator_space_name}_{str(timestamp)}"

    print("##### ("+search_operator_space_name+") determine_heuristic_space")

    heuristic_space = component_config.determine_heuristic_space(search_operator_space_path)

    # Configure Hyperheurstic Object.
    hh_parameters = experiment_config.tryGet('hh_parameters')
    timestamp = int(time.time())
    nr_of_iterations = experiment_config.tryGet('hh_parameters', 'num_iterations')
    nr_of_steps = experiment_config.tryGet('hh_parameters', 'num_steps')
    file_label = f"ASML-Faezeh_{experiment_name}_{nr_of_iterations}_iterations_{nr_of_steps}_steps_{str(timestamp)}"

    probs = {}
    num_replicas = hh_parameters["num_replicas"] if hh_parameters["num_replicas"] > 0 else 1 
    simulation_model_template_path = conf.tryGet("simulation_model", "simulation_model_paths", "simulation_model_template_path")
    template_ini_file_path = os.path.join(simulation_model_template_path, "platform.xml")
    

    for rep in range(num_replicas):
        probs[rep] = component_config.create_problem_instanceASML(template_ini_file_path, max_wfpm_runtime, min_wfpm_runtime, min_cost, max_cost, heur_sim_coordinator.simulation_run)
        probs[rep]['set_file_name_fitness_values']("fitness_values_"+str(search_operator_space_name)+"_replica_"+str(rep)+".json")

    hyp = hh.Hyperheuristic(
        heuristic_space=heuristic_space,
        problems=probs,
        parameters=hh_parameters,
        file_label=file_label,
        pass_finalised_positions=pass_finalised_positions,
        file_details= {
            "experiment_name": experiment_name,
            "hh_parameters": hh_parameters,
            "timestamp": timestamp,
            "search_operator_space_name": search_operator_space_name,
            "template_ini_file_path": template_ini_file_path,
            "coordinator_config": conf
        }
    )


    # Start timer for the heuristic run.
    start_time = time.time()


    # Start hyper-heuristic run.
    best_sol, best_perf, hist_curr, hist_best = hyp.solve()

    # End timer for the heuristic run.
    end_time = time.time()

    hh_run_meta_data = collect_data.calculate_distinct_simulation_components(start_time, end_time, heur_sim_coordinator)

    # Save the heuristic run data.
    save_run_path = os.path.join(os.getcwd(), "data/raw/results/experiment_1/", experiment_name)
    collect_data.save_hh_run_meta_data(save_run_path, best_sol, best_perf, hist_curr, hist_best, hh_run_meta_data)

    print(f" ("+search_operator_space_name+") Best solution: "+str(best_sol))
    print(f" ("+search_operator_space_name+") Best performance: "+str(best_perf))
    print(f" ("+search_operator_space_name+") Best history: "+str(hist_best))
    print(f" ("+search_operator_space_name+") Current history: "+str(hist_curr))

    save_runs.append({
        "experiment_name": experiment_name,
        "best_solution": best_sol,
        "best_performance": best_perf,
        "current_history": hist_curr,
        "best_history": hist_best
    })
