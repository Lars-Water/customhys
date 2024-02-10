import os
import numpy as np
import datetime
import time
from pathlib import Path

from tools.config_reader import Config
import tools.tools as tools
import tools.coordinator as coordinator
import models.model as model
from visualization import visualization

from data import collect_data

# from external.CUSTOMHys.customhys import metaheuristic as mh
# from external.CUSTOMHys.customhys import hyperheuristic as hh

from customhys import metaheuristic as mh
# from customhys import hyperheuristic as hh


'''
    Run the metaheuristic search operator.

    Args:
        heuristic_run_config (Config): The configuration object for the heuristic run.
        heur_sim_coordinator (HeuristicSimulationCoordinator): The coordinator object for the heuristic simulation workflow.
        heuristics_collection (list): The collection of heuristics to be used in the metaheuristic.
        num_agents (int): The number of agents in the metaheuristic.
        num_iterations (int): The number of iterations for the metaheuristic.
        verbose (bool): The flag to indicate if the metaheuristic should run in verbose mode.
'''
def run_mh(heuristic_run_config, heur_sim_coordinator, heuristics_collection, num_agents, num_iterations, verbose=False):

    # Run the metaheuristic for every specified number of backbones.
    runs_nr_of_backbones = heuristic_run_config.tryGet("runs_nr_of_backbones")
    for run in runs_nr_of_backbones:

        # Start timer for the heuristic run.
        start_time = time.time()

        # Create a problem instance for the metaheuristic.
        prob = create_mh_instance(run, heur_sim_coordinator)

        # Generate a metaheuristic search method for the CQN model.
        met = mh.Metaheuristic(prob, heuristics_collection, num_agents=num_agents, num_iterations=num_iterations)
        met.verbose = verbose

        # Reset the execution times before starting the heuristic run.
        heur_sim_coordinator.coordinator_execution_time = 0
        heur_sim_coordinator.simulation_execution_time = 0

        # Run the metaheuristic on the problem. The fitness value is calculated at every iteration step by the run function in the SimulationModel class.
        met.run()

        # End timer for the heuristic run.
        end_time = time.time()

        heuristic_run_meta_data = calculate_distinct_simulation_components(start_time, end_time, heur_sim_coordinator)

        # Save the heuristic run data.
        # TODO: Make the following variable configurable.
        collect_data.collect_mh_run(
            met,
            f"/home/larry/hyper-heuristic-dse-2.0/data/raw/results/metaheuristic/{heuristic_run_config.tryGet('mh_run_file_name')}/",
            run,
            num_agents,
            num_iterations,
            heuristic_run_meta_data
        )

    return


'''
    Run the hyperheuristic search operator.

    Args:
        heur_sim_coordinator (HeuristicSimulationCoordinator): The coordinator object for the heuristic simulation workflow.
        heuristics_collection (list): The collection of heuristics to be used in the hyperheuristic.
'''
def run_hh(heur_sim_coordinator, heuristics_collection):

    # Setup parameters for the hyperheuristic
    hh_parameters = dict(
    cardinality=3,  # Max. numb. of SOs in MHs, lvl:1
    cardinality_min=1,  # Min. numb. of SOs in MHs, lvl:1
    num_iterations=100,  # Iterations an MH performs, lvl:1
    num_agents=30,  # Agents in population,     lvl:1
    as_mh=True,  # HH sequence as a MH?,     lvl:2
    num_replicas=30,  # Replicas per each MH,     lvl:2
    num_steps=50,  # Trials per HH step,       lvl:2
    stagnation_percentage=0.37,  # Stagnation percentage,    lvl:2
    max_temperature=100,  # Initial temperature (SA), lvl:2
    min_temperature=1e-6,  # Min temperature (SA),     lvl:2
    cooling_rate=1e-3,  # Cooling rate (SA),        lvl:2
    temperature_scheme='fast',  # Temperature updating (SA),lvl:2
    acceptance_scheme='exponential',  # Acceptance mode,          lvl:2
    allow_weight_matrix=True,  # Weight matrix,            lvl:2
    trial_overflow=False,  # Trial overflow policy,    lvl:2
    learnt_dataset=None,  # If it is a learnt dataset related with the heuristic space
    repeat_operators=True,  # Allow repeating SOs inSeq,lvl:2
    verbose=True,  # Verbose process,          lvl:2
    learning_portion=0.37,  # Percent of seqs to learn  lvl:2
    solver='static')  # Indicate which solver use lvl:1

    for run in [6]:
        subnet_structure = dict()
        subnet_structure["boundaries"] = {
                f"cable_{i}": [0, 1] for i in range(1, run + 1)
            }
        subnet_structure["optimal_solution"] = [0.9] * run,
        subnet_structure["optimal_fitness"] = 30

        cqn_instance = model.generate_instance(
            run,
            subnet_structure,
            heur_sim_coordinator.simulation_run
        )
        prob = cqn_instance.get_formatted_problem()

        # Create hyperheuristic object and solve the problem.
        hyp = hh.Hyperheuristic(
            heuristic_space=heuristics_collection,
            problem=prob,
            parameters=hh_parameters,
            file_label="INET-LANS"
        )

        best_sol, best_perf, hist_curr, hist_best = hyp.solve()

        print(f"Best solution: {best_sol}")
        print(f"Best performance: {best_perf}")
        print(f"Current history: {hist_curr}")
        print(f"Best history: {hist_best}")

        return

'''
    Collect the simulation stats from the simulation run.

    Args:
        heur_sim_coordinator (HeuristicSimulationCoordinator): The coordinator object for the heuristic simulation workflow.
        uid (str): The unique identifier for the simulation run.
'''
def process_simulated_instance(heur_sim_coordinator, uid):
    heur_sim_coordinator.transform_scalar_files(uid)
    simulation_metrics = heur_sim_coordinator.obtain_simulation_stats(uid)
    fitness_value = heur_sim_coordinator.fitness_evaluation(simulation_metrics)

    return fitness_value


'''
    Calculate the distinct components of the heuristic run.

    Args:
        start_time (float): The start time of the heuristic run.
        end_time (float): The end time of the heuristic run.
        heur_sim_coordinator (HeuristicSimulationCoordinator): The coordinator object for the heuristic simulation workflow.

    Returns:
        dict: The distinct components of the heuristic run.
'''
def calculate_distinct_simulation_components(start_time, end_time, heur_sim_coordinator):

    # Caculate the distinct execution times for the distinct components of the heuristic run.
    total_execution_time = end_time - start_time
    heuristic_execution_time = total_execution_time - heur_sim_coordinator.coordinator_execution_time
    coordinator_execution_time = heur_sim_coordinator.coordinator_execution_time - heur_sim_coordinator.simulation_execution_time
    simulation_execution_time = heur_sim_coordinator.simulation_execution_time

    return {
        "total_execution_time": total_execution_time,
        "heuristic_execution_time": heuristic_execution_time,
        "coordinator_execution_time": coordinator_execution_time,
        "simulation_execution_time": simulation_execution_time
    }


'''
    Define the heuristic operators to be used in the metaheuristic.

    Returns:
        list: The collection of heuristic operators to be used in the metaheuristic.
'''
def define_heuristic_operators(heuristic_run_config):
    heursitic_operators = heuristic_run_config.tryGet("metaheuristic")
    return [
        (
            heursitic_operator["name"],
            heursitic_operator["params"],
            heursitic_operator["strategy"]
        )
        for heursitic_operator in heursitic_operators
    ]


'''
    Create a problem instance for the metaheuristic.

    Args:
        run (int): The number of backbones in the problem instance.
        heur_sim_coordinator (HeuristicSimulationCoordinator): The coordinator object for the heuristic simulation workflow.
    Returns:
        dict: The problem instance for the metaheuristic.
'''
def create_mh_instance(run, heur_sim_coordinator):
    subnet_structure = dict()

    # Create a formulation of the problem instance.
    subnet_structure["boundaries"] = {
            f"cable_{i}": [0, 1] for i in range(1, run + 1)
        }
    subnet_structure["optimal_solution"] = [0.9] * run,
    subnet_structure["optimal_fitness"] = 30

    # Create a problem instance from the formulation.
    problem_instance = model.generate_instance(
        run,
        subnet_structure,
        heur_sim_coordinator.simulation_run
    )
    return problem_instance.get_formatted_problem()


'''
    Run the heuristic simulation workflow.

    Args:
        base_path (str): The base path for the project.
'''
def main(base_path):

    # Set up the general heuristic run configuration.
    # TODO: Make the following two variables configurable.
    heuristic_run_config_file = os.path.join(base_path, "config/config_heuristic_run.json")
    # TODO: Move folder 'heuristic' to a generalized logs folder.
    heuristic_run_path = os.path.join(base_path, "heuristic")
    heuristic_run_config = Config(Path(heuristic_run_config_file), Path(heuristic_run_path), "heuristic_run_config_manager")

    # Obtain the configuration parameters for the heuristic run.
    nr_of_agents = heuristic_run_config.tryGet("nr_of_agents")
    nr_of_iterations = heuristic_run_config.tryGet("nr_of_iterations")

    # Create the coordinator for the heuristic simulation workflow.
    heur_sim_coordinator = coordinator.HeuristicSimulationCoordinator(nr_of_agents) # noqa 501

    # Define the heuristic operators to be used.
    heuristics_collection = define_heuristic_operators(heuristic_run_config)

    run_mh(heuristic_run_config, heur_sim_coordinator, heuristics_collection, nr_of_agents, nr_of_iterations, verbose=False)

    # run_hh(heur_sim_coordinator, heuristics_collection)

    heur_sim_coordinator.shutdown_manager()

    return


if __name__ == "__main__":
    base_path = Path(os.getcwd()).parent
    main(base_path)

    # visualization.bar_execution_times(type="iterations")
    # visualization.combined_pie_execution_times(type="iterations")

    # visualization.plot_best_fitness_per_file()
    # visualization.plot_convergence_vs_components()
