import os
import time

from src_main.external.customhys.customhys import hyperheuristic as hh

from src_main.data import collect_data
from src_main.tools import component_config
from src_main.tools import coordinator


def run_experiment(experiment_config, coordinator_params):
    '''
        Experimental run of tuning the parameters of any provided search operators.
    '''
    print("##### exp_1_flow.run_experiment ")
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

    save_runs = []

    print("##### for search_operator_space_path ")
    for search_operator_space_path, search_operator_space_name in zip(search_operator_space_paths, search_operator_space_names):

        # Open a file in write mode ('w') and write the string
        with open("output.txt", "w") as file:
            file.write(f"Running {search_operator_space_path} - {search_operator_space_name}")

        # Get the current Unix timestamp
        timestamp = int(time.time())
        experiment_name = f"experiment_1_{search_operator_space_name}_{str(timestamp)}"

        print("##### determine_heuristic_space")

        heuristic_space = component_config.determine_heuristic_space(search_operator_space_path)

        # Configure Hyperheurstic Object.
        hh_parameters = experiment_config.tryGet('hh_parameters')
        timestamp = int(time.time())
        nr_of_iterations = experiment_config.tryGet('hh_parameters', 'num_iterations')
        nr_of_steps = experiment_config.tryGet('hh_parameters', 'num_steps')
        file_label = f"INET-LANS_{experiment_name}_{nr_of_iterations}_iterations_{nr_of_steps}_steps_{str(timestamp)}_{nr_of_backbone_switches}_switches"
        hyp = hh.Hyperheuristic(
            heuristic_space=heuristic_space,
            problem=prob,
            parameters=hh_parameters,
            file_label=file_label,
            pass_finalised_positions=pass_finalised_positions
        )

        print("##### start_time")
        # Start timer for the heuristic run.
        start_time = time.time()

        # Reset the execution times before starting the heuristic run.
        heur_sim_coordinator.coordinator_and_simulation_execution_time = 0
        heur_sim_coordinator.simulation_execution_time = 0

        # Start hyper-heuristic run.
        best_sol, best_perf, hist_curr, hist_best = hyp.solve()

        # End timer for the heuristic run.
        end_time = time.time()

        print("###### hh_run_meta_data")
        hh_run_meta_data = collect_data.calculate_distinct_simulation_components(start_time, end_time, heur_sim_coordinator)

        print("###### save_run_path")
        # Save the heuristic run data.
        save_run_path = os.path.join(os.getcwd(), "data/raw/results/experiment_1/", search_operator_space_name)
        collect_data.save_hh_run_meta_data(save_run_path, best_sol, best_perf, hist_curr, hist_best, hh_run_meta_data)

        print(f"Best solution: {best_sol}")
        print(f"Best performance: {best_perf}")
        print(f"Current history: {hist_curr}")
        print(f"Best history: {hist_best}")

        save_runs.append({
            "experiment_name": experiment_name,
            "best_solution": best_sol,
            "best_performance": best_perf,
            "current_history": hist_curr,
            "best_history": hist_best
        })