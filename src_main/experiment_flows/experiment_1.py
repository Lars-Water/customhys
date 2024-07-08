import os
import time

from customhys import hyperheuristic as hh

import src_main.models.model as model
from src_main.data import collect_data
from src_main.tools import component_config


def determine_heuristic_space(search_operator_space_path):
    # Open the file for reading
    with open(search_operator_space_path, 'r') as file:
        # Read all lines from the file
        lines = file.readlines()

    # Initialize an empty list to store the tuples
    heurstic_space = []

    # Iterate over each line in the file
    for line in lines:
        # Strip any leading/trailing whitespace from the line
        line = line.strip()
        # Use eval to convert the string representation of the tuple into an actual tuple
        tuple_item = eval(line)
        # Append the tuple to the list
        heurstic_space.append(tuple_item)

    # Print the list of tuples
    return heurstic_space


def run_experiment(experiment_config, heur_sim_coordinator, nr_of_backbone_switches):
    '''
        Experimental run of tuning the parameters of any provided search operators.
    '''
    search_operator_space_paths = experiment_config.tryGet('search_operator_space_paths')
    search_operator_space_names = experiment_config.tryGet('search_operator_space_names')

    # Create problem instance.
    min_datarate, max_datarate, min_cost, max_cost = heur_sim_coordinator.get_datarate_cost_boundaries()
    prob = component_config.create_problem_instance(nr_of_backbone_switches, max_datarate, min_datarate, max_cost, min_cost, heur_sim_coordinator.simulation_run)

    save_runs = []

    for search_operator_space_path, search_operator_space_name in zip(search_operator_space_paths, search_operator_space_names):
        # Get the current Unix timestamp
        timestamp = int(time.time())
        experiment_name = f"experiment_1_{search_operator_space_name}_{str(timestamp)}"

        heuristic_space = determine_heuristic_space(search_operator_space_path)

        # Configure Hyperheurstic Object.
        hh_parameters = experiment_config.tryGet('hh_parameters')
        hyp = hh.Hyperheuristic(
            heuristic_space=heuristic_space,
            problem=prob,
            parameters=hh_parameters,
            file_label="INET-LANS_" + experiment_name
        )

        # Start timer for the heuristic run.
        start_time = time.time()

        # Reset the execution times before starting the heuristic run.
        heur_sim_coordinator.coordinator_and_simulation_execution_time = 0
        heur_sim_coordinator.simulation_execution_time = 0

        # Start hyper-heuristic run.
        best_sol, best_perf, hist_curr, hist_best = hyp.solve()

        # End timer for the heuristic run.
        end_time = time.time()

        hh_run_meta_data = collect_data.calculate_distinct_simulation_components(start_time, end_time, heur_sim_coordinator)

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
