import os
import time
from pathlib import Path
import argparse

from src_main.tools.config_reader import Config
import src_main.tools.coordinator as coordinator
import src_main.models.model as model
from src_main.data import collect_data
from src_main.visualization import visualization

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
        base_path (string): Root of the project.
        verbose (bool): The flag to indicate if the metaheuristic should run in verbose mode.
'''
def run_mh(heuristic_run_config, heur_sim_coordinator, heuristics_collection, num_agents, num_iterations, base_path, verbose=False):

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
        save_run_path = os.path.join(base_path, "data/raw/results/metaheuristic/", heuristic_run_config.tryGet('mh_save_run_path_mh_name'))
        collect_data.collect_mh_run(met,save_run_path,run,num_agents,num_iterations,heuristic_run_meta_data)

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
        coordinator_config_file_path (Path): The path to the coordinator configuration file.
        heur_run_config_file_path (Path): The path to the heuristic run configuration file.
        parameter_tuning (bool): The flag to indicate if parameter tuning is enabled.
'''
def main(base_path, coordinator_config_file_path, heur_run_config_file_path, parameter_tuning):

    # Set up the general heuristic run configuration.
    if heur_run_config_file_path:
        heuristic_run_log_path = os.path.join(base_path, "data/logs/heuristic_run/")
        os.makedirs(heuristic_run_log_path, exist_ok=True)
        heuristic_run_config = Config(heur_run_config_file_path, Path(heuristic_run_log_path), "heuristic_run_config_manager")

        # Obtain the configuration parameters for the heuristic run.
        nr_of_agents = heuristic_run_config.tryGet("nr_of_agents")
        nr_of_iterations = heuristic_run_config.tryGet("nr_of_iterations")

        # Define the heuristic operators to be used.
        heuristics_collection = define_heuristic_operators(heuristic_run_config)

        # Create the coordinator for the heuristic simulation workflow.
        heur_sim_coordinator = coordinator.HeuristicSimulationCoordinator(base_path, coordinator_config_file_path, nr_of_agents) # noqa 501

        run_mh(heuristic_run_config, heur_sim_coordinator, heuristics_collection, nr_of_agents, nr_of_iterations, base_path, verbose=False)
        # run_hh(heur_sim_coordinator, heuristics_collection)
    else:
        nr_of_agents = None
        parameter_tuning_coordinator = coordinator.HeuristicSimulationCoordinator(base_path, coordinator_config_file_path, nr_of_agents, True) # noqa 501
        parameter_tuning_coordinator.parameter_tuning_workflow()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the heuristic simulation workflow.")
    parser.add_argument("--base_path", type=str, required=True, help="Root of the project.")
    parser.add_argument("--coordinator_config", type=str, required=True, help="Path to the coordinator configuration file.")
    parser.add_argument("--heur_run_config", type=str, required=False, help="Path to the heuristic run configuration file.")
    parser.add_argument("--param_tune", action="store_true", required=False, help="Flag that enables parameter tuning.")
    args = parser.parse_args()

    # Convert the arguments to Path objects.
    base_path = Path(args.base_path)
    coordinator_config_file_path = Path(args.coordinator_config)
    heur_run_config_file_path = Path(args.heur_run_config) if args.heur_run_config else None
    parameter_tuning = args.param_tune

    # Check if the base path exists.
    if not base_path.exists():
        raise FileNotFoundError(f"The base path {base_path} does not exist.")

    # Check if the coordinator configuration file exists
    if not coordinator_config_file_path.exists():
        raise FileNotFoundError(f"The coordinator configuration file {coordinator_config_file_path} does not exist.")

    # Check if the heuristic run configuration file exists
    if not parameter_tuning and not heur_run_config_file_path.exists():
        raise FileNotFoundError(f"The heuristic run configuration file {heur_run_config_file_path} does not exist.")
    elif not parameter_tuning:
        raise ValueError("No functionality called for experiment. Either provide a heuristic run config or enable parameter tuning.")

    main(base_path, coordinator_config_file_path, heur_run_config_file_path, parameter_tuning)

    # visualization.bar_execution_times(type="iterations")
    # visualization.combined_pie_execution_times(type="iterations")

    # visualization.plot_best_fitness_per_file()
    # visualization.plot_convergence_vs_components()
