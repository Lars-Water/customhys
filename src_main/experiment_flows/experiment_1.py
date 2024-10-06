import os
import time
# from multiprocessing import Process
import threading

from src_main.external.customhys.customhys import hyperheuristic as hh

from src_main.data import collect_data
from src_main.tools import component_config
from src_main.tools import coordinator

save_runs = []
experiment_config = None
def run_experiment(experiment_config, coordinator_params):
    '''
        Experimental run of tuning the parameters of any provided search operators.
    '''
    print("##### exp_1_flow.run_experiment ")
    np.seterr(divide='ignore', invalid='ignore')
    search_operator_space_paths = experiment_config.tryGet('search_operator_space_paths')
    search_operator_space_names = experiment_config.tryGet('search_operator_space_names')
    pass_finalised_positions = experiment_config.tryGet('pass_finalised_positions')

    # Create the HeuristicSimulationCoordinator.
    base_path, coordinator_config_file_path, nr_of_agents, run_name, nr_of_backbone_switches = coordinator_params
    heur_sim_coordinator = coordinator.HeuristicSimulationCoordinator(base_path, coordinator_config_file_path, nr_of_agents, run_name, nr_of_backbone_switches) # noqa 501

    # Create problem instance.
    print("##### manual_normalization ")
    heur_sim_coordinator.manual_normalization()
    min_datarate, max_datarate, min_cost, max_cost = heur_sim_coordinator.get_datarate_cost_boundaries()
    print("##### prob ")
    prob = component_config.create_problem_instance(nr_of_backbone_switches, max_datarate, min_datarate, max_cost, min_cost, heur_sim_coordinator.simulation_run)

    # Reset the execution times before starting the heuristic run.
    heur_sim_coordinator.coordinator_and_simulation_execution_time = 0
    heur_sim_coordinator.simulation_execution_time = 0
    

    print("### for search_operator_space_path ")
    fns_hh = []
    for search_operator_space_path, search_operator_space_name in zip(search_operator_space_paths, search_operator_space_names):
        fns_hh.append(threading.Thread(target=run_search_operator_space_path, args=(
    experiment_config, search_operator_space_path, search_operator_space_name, nr_of_backbone_switches, prob, heur_sim_coordinator)))


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

def run_search_operator_space_path(experiment_config, search_operator_space_path, search_operator_space_name, nr_of_backbone_switches, prob, heur_sim_coordinator):
    pass_finalised_positions = experiment_config.tryGet('pass_finalised_positions')
        # Open a file in write mode ('w') and write the string
    with open("output.txt", "a") as file:
        file.write(f"Running {search_operator_space_path} - {search_operator_space_name}")

    # Get the current Unix timestamp
    timestamp = int(time.time())
    experiment_name = f"experiment_1_{search_operator_space_name}_{str(timestamp)}"

    print("##### ("+search_operator_space_name+") determine_heuristic_space")

    heuristic_space = component_config.determine_heuristic_space(search_operator_space_path)

    # Configure Hyperheurstic Object.
    hh_parameters = experiment_config.tryGet('hh_parameters')
    timestamp = int(time.time())
    nr_of_iterations = experiment_config.tryGet('hh_parameters', 'num_iterations')
    nr_of_steps = experiment_config.tryGet('hh_parameters', 'num_steps')
    file_label = f"INET-LANS_{experiment_name}_{nr_of_iterations}_iterations_{nr_of_steps}_steps_{str(timestamp)}_{nr_of_backbone_switches}_switches"
    prob_local = prob
    prob_local['set_file_name_fitness_values']("fitness_values_"+str(search_operator_space_name)+".json")
    hyp = hh.Hyperheuristic(
        heuristic_space=heuristic_space,
        problem=prob_local,
        parameters=hh_parameters,
        file_label=file_label,
        pass_finalised_positions=pass_finalised_positions,
        file_name_fitness_values="fitness_values_"+str(search_operator_space_name)+".json"
    )


    # Start timer for the heuristic run.
    start_time = time.time()


    # Start hyper-heuristic run.
    best_sol, best_perf, hist_curr, hist_best = hyp.solve()

    # End timer for the heuristic run.
    end_time = time.time()

    hh_run_meta_data = collect_data.calculate_distinct_simulation_components(start_time, end_time, heur_sim_coordinator)

    # Save the heuristic run data.
    save_run_path = os.path.join(os.getcwd(), "data/raw/results/experiment_1/", search_operator_space_name)
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
