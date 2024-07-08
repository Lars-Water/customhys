import time
import os
from pathlib import Path

from customhys import hyperheuristic as hh
from customhys import metaheuristic as mh

from src_main.tools import component_config
from src_main.tools.config_reader import Config
from src_main.tools import coordinator

from src_main.data import collect_data


def _determine_metaheurstic_name(metaheuristic_path):
    return os.path.splitext(os.path.basename(metaheuristic_path))[0]


def create_hh():
    pass


def run_experiment(base_path, experiment_config, coordinator_config_file_path):

    # Define the number of agents and iterations for the heuristics.
    nr_of_agents = experiment_config.tryGet('nr_of_agents')
    nr_of_iterations = experiment_config.tryGet('nr_of_iterations')
    nrs_of_backbones = experiment_config.tryGet('nr_of_backbones')

    for nr_of_backbones in nrs_of_backbones:

        # TODO: Remove dependency of coordinator functionality based on the number of backbones.
        # Create the coordinator for the corresponding backbone number.
        run_name = f"experiment_2_{str(nr_of_backbones)}_backbones"
        heur_sim_coordinator = coordinator.HeuristicSimulationCoordinator(base_path, coordinator_config_file_path, nr_of_agents, run_name, nr_of_backbones) # noqa 501

        # Create problem instance.
        min_datarate, max_datarate, min_cost, max_cost = heur_sim_coordinator.get_datarate_cost_boundaries()
        prob = component_config.create_problem_instance(nr_of_backbones, max_datarate, min_datarate, max_cost, min_cost, heur_sim_coordinator.simulation_run)

        # Create the metaheuristics.
        base_path = os.getcwd()
        metaheuristic_paths = experiment_config.tryGet('metaheuristics_paths')
        for metaheuristic_path in metaheuristic_paths:
            metaheuristic_name = _determine_metaheurstic_name(metaheuristic_path)
            met = component_config.create_mh(metaheuristic_path, metaheuristic_name, nr_of_agents, nr_of_iterations, prob, base_path, verbose=False)

            # Reset the execution times before starting the heuristic run.
            heur_sim_coordinator.coordinator_and_simulation_execution_time = 0
            heur_sim_coordinator.simulation_execution_time = 0

            # Run the metaheuristic on the problem. The fitness value is calculated at every iteration step by the run function in the SimulationModel class.
            start_time = time.time()
            met.run()
            end_time = time.time()

            heuristic_run_meta_data = collect_data.calculate_distinct_simulation_components(start_time, end_time, heur_sim_coordinator)

            # Save the heuristic run data.
            save_run_path = os.path.join(base_path, "data/raw/results/experiment_2/metaheuristics/", metaheuristic_name)
            collect_data.collect_mh_run(met, save_run_path, nr_of_backbones, nr_of_agents, nr_of_iterations, heuristic_run_meta_data)
