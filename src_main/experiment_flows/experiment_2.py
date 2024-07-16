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


def _run_mh(metaheuristic_path, nr_of_agents, nr_of_iterations, prob, base_path, heur_sim_coordinator, nr_of_backbones):
    metaheuristic_name = _determine_metaheurstic_name(metaheuristic_path)
    met = component_config.create_mh(metaheuristic_path, metaheuristic_name, nr_of_agents, nr_of_iterations, prob, base_path, verbose=False)

    # Reset the execution times before starting the heuristic run.
    heur_sim_coordinator.coordinator_and_simulation_execution_time = 0
    heur_sim_coordinator.simulation_execution_time = 0

    # Set the run name for the heuristic run.
    run_name = f"experiment_2_{metaheuristic_name}_{str(nr_of_backbones)}_backbones"
    heur_sim_coordinator.set_run_name(run_name)

    # Run the metaheuristic on the problem. The fitness value is calculated at every iteration step by the run function in the SimulationModel class.
    start_time = time.time()
    met.run()
    end_time = time.time()

    heuristic_run_meta_data = collect_data.calculate_distinct_simulation_components(start_time, end_time, heur_sim_coordinator)

    # Save the heuristic run data.
    save_run_path = os.path.join(base_path, "data/raw/results/experiment_2/metaheuristics/", metaheuristic_name)
    collect_data.collect_mh_run(met, save_run_path, nr_of_backbones, nr_of_agents, nr_of_iterations, heuristic_run_meta_data)


def _run_hh(experiment_config, prob, heur_sim_coordinator, nr_of_backbones):
    # Run the Hyperheuristic of the metaheuristics run above.
    search_operator_space_path = experiment_config.tryGet('search_operator_space_path')
    search_operator_space_name = experiment_config.tryGet('search_operator_space_name')

    # Define experiment name.
    timestamp = int(time.time())
    experiment_name = f"experiment_2_{nr_of_backbones}_{search_operator_space_name}_{str(timestamp)}"

    # Determine the hyperheuristic search space.
    heuristic_space = component_config.determine_heuristic_space(search_operator_space_path)

    # Configure Hyperheurstic.
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
    save_run_path = os.path.join(os.getcwd(), "data/raw/results/experiment_2/", f"hh_{search_operator_space_name}")
    collect_data.save_hh_run_meta_data(save_run_path, best_sol, best_perf, hist_curr, hist_best, hh_run_meta_data)

    print(f"Best solution: {best_sol}")
    print(f"Best performance: {best_perf}")
    print(f"Current history: {hist_curr}")
    print(f"Best history: {hist_best}")


def run_experiment(base_path, experiment_config, coordinator_config_file_path):

    # Define the number of agents and iterations for the coordinator and metaheuristics.
    nr_of_agents = experiment_config.tryGet('nr_of_agents')
    nr_of_iterations = experiment_config.tryGet('nr_of_iterations')
    nrs_of_backbones = experiment_config.tryGet('nr_of_backbones')
    heur_sim_coordinator = coordinator.HeuristicSimulationCoordinator(base_path, coordinator_config_file_path, nr_of_agents) # noqa 501

    for nr_of_backbones in nrs_of_backbones:

        # Set the number of backbones accordingly.
        heur_sim_coordinator.set_nr_of_backbone_switches(nr_of_backbones)

        # Create problem instance.
        min_datarate, max_datarate, min_cost, max_cost = heur_sim_coordinator.get_datarate_cost_boundaries()
        prob = component_config.create_problem_instance(nr_of_backbones, max_datarate, min_datarate, max_cost, min_cost, heur_sim_coordinator.simulation_run)

        # # Run the metaheuristics.
        # base_path = os.getcwd()
        # metaheuristic_paths = experiment_config.tryGet('metaheuristics_paths')
        # for metaheuristic_path in metaheuristic_paths:
        #     _run_mh(metaheuristic_path, nr_of_agents, nr_of_iterations, prob, base_path, heur_sim_coordinator, nr_of_backbones)

        # Run the hyperheuristic.
        _run_hh(experiment_config, prob, heur_sim_coordinator, nr_of_backbones)
