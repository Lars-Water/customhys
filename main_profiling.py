import matplotlib.pyplot as plt
import json
import pandas as pd
import math
import re
import customhys
import sys

import warnings
import os
import time
from pathlib import Path
import argparse

# Yappi imports
import threading
import yappi

from src_main.tools.config_reader import Config
import src_main.tools.coordinator_inet as coordinator_inet
from src_main.data import collect_data_INET
import src_main.experiment_flows.experiment_1 as exp_1_flow
import src_main.experiment_flows.experiment_2 as exp_2_flow
import src_main.experiment_flows.experiment_asml as exp_asml_flow
from src_main.visualization import visualization
from src_main.tools import file_operations as fo
from src_main.tools import component_config as cc
import logging
from src_main.tools.logger import logger as logger_main, setLevelLogger, loggerRICH


from setuptools import setup, find_packages
from pathlib import Path


sys.path.insert(0, Path(__file__).parent.as_posix())
sys.path.append("./src_main/external/simulation_model")


import shutil
from datetime import datetime


#     def print_all(
    #     self,
    #     out=sys.stdout,
    #     columns={
    #         0: ("name", 36),
    #         1: ("ncall", 5),
    #         2: ("tsub", 8),
    #         3: ("ttot", 8),
    #         4: ("tavg", 8)
    #     }
    # ):


class LoggerWriter:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message):
        # Accumulate until a newline is seen, then log each full line
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self.logger.log(self.level, line)

    def flush(self):
        # If anything is left without a newline, log it too
        if self._buffer:
            self.logger.log(self.level, self._buffer)
            self._buffer = ""
# ------------------------------------------------------------------------------
# Yappi Profiling Functions
# ------------------------------------------------------------------------------
def dump_yappi_stats(interval_seconds, output_dir):
    """
    Every 'interval_seconds' seconds, dump out current Yappi stats to a .pstat file.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger = logger_main("dump_yappi_stats", output_dir, disabled=False)
    setLevelLogger(logger, "DEBUG")

    counter = 0
    while True:
        logger.info(f"Dumping Yappi stats.")    
        time.sleep(interval_seconds)

        # Collect the stats object
        func_stats = yappi.get_func_stats()
        thread_stats = yappi.get_thread_stats()

        lw = LoggerWriter(logger, level=logging.INFO)
        # func_stats.print_all()
        thread_stats.print_all(out=lw)

        # Build a filename that reflects epoch or a counter
        timestamp = int(time.time())
        fname_func_stats = os.path.join(output_dir, f"yappi_snapshot_func_stats_{counter}.pstat")
        fname_thread_stats = os.path.join(output_dir, f"yappi_snapshot_thread_stats_{counter}.pstat")

        # Save in pstat (compatible with pstats / SnakeViz)
        threads = yappi.get_thread_stats()
        for thread in threads:
            logger.info(f"Function stats for ({thread.name}) ({thread.id})")
            fname_func_stats_thread = os.path.join(output_dir, f"yappi_snapshot_func_stats_{counter}_{thread.name}_{thread.id}.pstat")
            yappi.get_func_stats(ctx_id=thread.id).save(fname_func_stats_thread, type="pstat")

        func_stats.save(fname_func_stats, type="pstat")
        yappi.clear_stats()
        # thread_stats.save(fname_thread_stats, type="text")

        # yappi.get_func_stats().sort(sort_type='totaltime', sort_order='desc').debug_print()
        logger.info(f"[INFO] Yappi stats saved to {fname_func_stats}, {fname_thread_stats}")

        # Optionally, also reset stats so that next interval is "fresh"
        # yappi.clear_stats() # Uncomment if windowed profiling is preferred

        counter += 1

def start_periodic_profiling(interval_seconds=300,  # e.g. every 5 min
                             output_subdir="yappi_profiles"):
    """
    Call this before you kick off the main hyper-heuristic loops.
    It:
      1) starts Yappi in wall-clock mode (measures real time, not just CPU),
      2) launches a daemon thread that sleeps for `interval_seconds` and then
         writes out a .pstat file under the generated output directory.
    """
    # 1) Determine a unique directory for this profiling session
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    
    if slurm_job_id:
        # If running under a SLURM job, use the job ID for the subdirectory name
        unique_subdir_name = f"job_{slurm_job_id}"
    else:
        # Not under SLURM or SLURM_JOB_ID not set, create a unique dir name with a timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f") # %f for microseconds
        unique_subdir_name = f"run_{timestamp_str}"
        
    this_output_dir = os.path.join(output_subdir, unique_subdir_name)
    os.makedirs(this_output_dir, exist_ok=True) # Creates the full path including output_subdir if needed

    # 3) Start Yappi
    #    Wall-clock mode is recommended if your per-iteration may include I/O or sleeps
    yappi.set_clock_type("wall") 
    yappi.start()

    # 4) Launch the dumper thread (daemon so it will not block program exit)
    dumper = threading.Thread(
        target=dump_yappi_stats,
        args=(interval_seconds, this_output_dir),
        daemon=True
    )
    dumper.start()

    # Return the path so logging can reference it
    return this_output_dir
# ------------------------------------------------------------------------------

def remove_directory(directory_path):
    '''
        Quick and dirty solution for running many heuristic runs after each other without clogging up memory.
    '''
    try:
        shutil.rmtree(directory_path)  # Remove the entire directory and all its contents
        print(f'Successfully removed {directory_path}')
    except Exception as e:
        print(f'Failed to delete {directory_path}. Reason: {e}')


def experiment_asml(base_path, coordinator_config_file_path):
    logger.info(f"Experiment ASML started.")    
    conf = Config(coordinator_config_file_path, name = "main_conf_logPath")
    log_path = conf.tryGet("output_paths", "log_files")
    if log_path is None:
        log_path = os.path.join(base_path, "data/logs/")

    experiment_1_config_file_path = Path(os.path.join(base_path, "config/experiment_asml_06_2025/experiment_asml_06_2025.json"))
    experiment_1_log_path = os.path.join(log_path, "experiments/experiment_asml_06_2025")

    config_manager_filename = f"experiment_asml_config_manager_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    experiment_1_config = Config(experiment_1_config_file_path, Path(experiment_1_log_path), config_manager_filename)
    # TODO: Change this naming flow, because it goes from main, to coordinator, to data collector.
    nr_of_agents = experiment_1_config.tryGet("hh_parameters", "num_agents")
    run_name = "experiment_asml"
    coordinator_params = (base_path, coordinator_config_file_path, nr_of_agents, run_name)

    # Run Experiment 1.
    logger.info(f"Experiment ASML running.")    
    exp_asml_flow.run_experiment(experiment_1_config, coordinator_params)

def experiment_1(base_path, coordinator_config_file_path, nr_of_backbone_switches):
    """
    Set up and run Experiment 1.
    Parameters:
    base_path (str): The base directory path where configuration and log files are stored.
    coordinator_config_file_path (str): The file path to the coordinator configuration file.
    nr_of_backbone_switches (int): The number of backbone switches to be used in the experiment.
    Returns:
    None
    """
    # Set up the experiment_1 configuration object.
    conf = Config(coordinator_config_file_path, name = "main_conf_logPath")
    log_path = conf.tryGet("output_paths", "log_files")
    if log_path is None:
        log_path = os.path.join(base_path, "data/logs/")

        
    experiment_1_config_file_path = Path(os.path.join(base_path, "config/experiment_1/experiment_1.json"))
    experiment_1_log_path = os.path.join(log_path, "experiments/experiment_1")
    config_manager_filename =f"experiment_1_config_manager_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    experiment_1_config = Config(experiment_1_config_file_path, Path(experiment_1_log_path), config_manager_filename)

    # TODO: Change this naming flow, because it goes from main, to coordinator, to data collector.
    nr_of_agents = experiment_1_config.tryGet("hh_parameters", "num_agents")
    run_name = "experiment_1"
    coordinator_params = (base_path, coordinator_config_file_path, nr_of_agents, run_name, nr_of_backbone_switches)

    # Run Experiment 1.
    exp_1_flow.run_experiment(experiment_1_config, coordinator_params)


def experiment_2(base_path, coordinator_config_file_path):
    """
    Set up and run Experiment 2.
    This function sets up the configuration for Experiment 2, creates necessary directories,
    and runs the experiment using the provided base path and coordinator configuration file path.
    Args:
        base_path (str): The base directory path where configuration and log files are located.
        coordinator_config_file_path (str): The file path to the coordinator configuration file.
    Returns:
        None
    """
    conf = Config(coordinator_config_file_path, name = "main_conf_logPath")
    log_path = conf.tryGet("output_paths", "log_files")
    if log_path is None:
        log_path = os.path.join(base_path, "data/logs/")

    # Set up the experiment_2 configuration object.
    experiment_2_config_file_path = Path(os.path.join(base_path, "config/experiment_2/experiment_2.json"))
    experiment_2_log_path = os.path.join(log_path, "experiments/experiment_2")

    config_manager_filename = f"experiment_2_config_manager_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    experiment_2_config = Config(experiment_2_config_file_path, Path(experiment_2_log_path), config_manager_filename)

    # Run Experiment 2.
    exp_2_flow.run_experiment(base_path, experiment_2_config, coordinator_config_file_path)

def create_ga_heurstic_space():
    search_operator_name="genetic_algorithm"
    experiment_folder = "experiment_1"
    possible_ga_default_param_values = {
        "pairing": ['cost', 'rank', 'tournament_2_100', 'tournament_2_75', 'tournament_2_50', 'tournament_3_100', 'tournament_3_75', 'tournament_3_50', 'random', 'even-odd'],
        "crossover": ['single', 'two', 'uniform', 'blend', 'linear_0.5_0.5'],
        "mating_pool_factor": [0.1, 0.3, 0.5, 0.7, 0.9],
        "scale": [0.01, 0.2575, 0.505, 0.7525, 1.0],
        "elite_rate": [0.0, 0.225, 0.45, 0.675, 0.9],
        "mutation_rate": [0.1, 0.3, 0.5, 0.7, 0.9],
        "distribution": ['uniform', 'gaussian', 'levy'],
        "selector": ['greedy', 'probabilistic', 'metropolis', 'all', 'none']
    }
    fo.generate_heuristic_space_cartesian_product(search_operator_name=search_operator_name, experiment_folder=experiment_folder, config=possible_ga_default_param_values)


def create_ga_heuristic_space_lhs():
    ga_parameter_bounds = {
        "pairing": (['cost', 'rank', 'tournament_2_100', 'tournament_2_75', 'tournament_2_50', 'tournament_3_100', 'tournament_3_75', 'tournament_3_50', 'random', 'even-odd']),
        "crossover": (['single', 'two', 'uniform', 'blend', 'linear_0.5_0.5']),
        "mating_pool_factor": (0.1, 0.9),
        "scale": (0.01, 1.0),
        "elite_rate": (0.0, 0.9),
        "mutation_rate": (0.1, 0.9),
        "distribution": (['uniform', 'gaussian', 'levy']),
        "selector": (['greedy', 'probabilistic', 'metropolis', 'all', 'none'])
    }
    ga_heuristic_space_samples = cc.lhs_mh_heuristic_space_generation(param_bounds=ga_parameter_bounds, n_samples=1000)
    fo.generate_heuristic_space_lhs_sampling(search_operator_name="genetic_algorithm", experiment_folder="experiment_1", samples=ga_heuristic_space_samples)


'''
    Run the heuristic simulation workflow.

    Args:
        base_path (str): The base path for the project.
        coordinator_config_file_path (Path): The path to the coordinator configuration file.
        heur_run_config_file_path (Path): The path to the heuristic run configuration file.
        experiment (str): The experiments that were requested to be run.
        parameter_tuning (bool): The flag to indicate if parameter tuning is enabled.
        design_space_plot (bool): The flag to indicate if design space determination is enabled.
        nr_of_backbone_switches (int): The number of backbone switches for the INET model.
'''
def main(base_path, coordinator_config_file_path, heur_run_config_file_path, experiment, visualize, metaheuristics, hh_run_dirs_exp_1, hh_run_dirs_exp_2, parameter_tuning, design_space_plot, nr_of_backbone_switches):
    logger.info(f"Running the main function.")
    print("##### Customhys version:" + str(customhys.__version__))
    # Run the requested experiments.
    if experiment == '1':
        experiment_1(base_path, coordinator_config_file_path, nr_of_backbone_switches)
    elif experiment == '2':
        experiment_2(base_path, coordinator_config_file_path)
    elif experiment == 'asml':
        logger.info(f"Try to run the experiment ASML.")    
        experiment_asml(base_path, coordinator_config_file_path)
    elif experiment == 'all':
        logger.info(f"Try to run the experiment 1 and 2.")
        experiment_1(base_path, coordinator_config_file_path, nr_of_backbone_switches)
        experiment_2(base_path, coordinator_config_file_path)

    if visualize == '1':
        visualize_experiment_1(hh_run_dirs_exp_1)
    elif visualize == '2':
        visualize_experiment_2(base_path, metaheuristics, hh_run_dirs_exp_2)
    elif visualize == 'all':
        visualize_experiment_1(hh_run_dirs_exp_1)
        visualize_experiment_2(base_path, metaheuristics, hh_run_dirs_exp_2)


    # # Set up the general heuristic run configuration.
    # if heur_run_config_file_path:
    #     nr_of_backbone_switches = 50
    #     nr_of_cable_types = 4
    #     hh_run_path = Path(os.path.join(os.getcwd(), "data_files/raw/INET-LANS_experiment_2_200_subsequent_positions_1721088383"))
    #     collect_data_INET.quick_and_dirty_save_hh_positions_to_xlsx(hh_run_path, nr_of_backbone_switches, nr_of_cable_types, rescale=True)

    #     heuristic_run_log_path = os.path.join(base_path, "data/logs/heuristic_run/")
    #     os.makedirs(heuristic_run_log_path, exist_ok=True)
    #     heuristic_run_config = Config(heur_run_config_file_path, Path(heuristic_run_log_path), "heuristic_run_config_manager")

    #     # Obtain the configuration parameters for the heuristic run.
    #     nr_of_agents = heuristic_run_config.tryGet("nr_of_agents")
    #     nr_of_iterations = heuristic_run_config.tryGet("nr_of_iterations")

    #     # Visualisation of metaheuristic runs.
    #     collect_data_INET.vizualize_mh_runs(nr_of_backbone_switches)

        # # TODO: Quick and dirty Manual removal of directories to prevent memory clogging.
        # experiments_directory_path = Path("/var/scratch/lvdwater/experiments/data/campaign_custom/n1_w1_s16")
        # remove_directory(experiments_directory_path)
        # sims_directory_path = Path("/var/scratch/lvdwater/sims")
        # remove_directory(sims_directory_path)

    # elif parameter_tuning:
    #     parameter_tuning_coordinator = coordinator_inet.HeuristicSimulationCoordinatorINET(base_path, coordinator_config_file_path, None, nr_of_backbone_switches) # noqa 501
    #     parameter_tuning_coordinator.parameter_tuning_workflow()
    # elif design_space_plot:
    #     design_space_coordinator = coordinator_inet.HeuristicSimulationCoordinatorINET(base_path, coordinator_config_file_path, None, nr_of_backbone_switches) # noqa 501
    #     design_space_coordinator.determine_design_space()

    return


if __name__ == "__main__":
    logger = loggerRICH("MAIN PROFILING")
    setLevelLogger(logger, "DEBUG")
    parser = argparse.ArgumentParser(description="Run the heuristic simulation workflow.")
    parser.add_argument('--experiment', choices=['1', '2', 'asml', 'all'], required=False, help='Choose which experiment to run')
    parser.add_argument('--visualize', choices=['1', '2', 'all'], required=False, help='Choose which experiment to visualize')
    parser.add_argument('--metaheuristics', nargs='+', required=False, help='List of metaheuristics to visualize. Checks the results folder of experiment 2 for the singular metaheuristic runs.')
    parser.add_argument("--hh_run_dirs_exp_1", nargs='+', required=False, help="List of HH run directories for Experiment 1.")
    parser.add_argument("--hh_run_dirs_exp_2", nargs='+', required=False, help="List of HH run directories for Experiment 2.")
    parser.add_argument("--nr_sw", type=int, required=False, help="The number of backbone switches.")
    parser.add_argument("--base_path", type=str, required=True, help="Root of the project.")
    parser.add_argument("--coordinator_config", type=str, required=True, help="Path to the coordinator configuration file.")
    parser.add_argument("--heur_run_config", type=str, required=False, help="Path to the heuristic run configuration file.")
    parser.add_argument("--param_tune", action="store_true", required=False, help="Flag that enables parameter tuning.")
    parser.add_argument("--design_space_plot", action="store_true", required=False, help="Flag that enables determining the design space.")

    logger.info("Running with profiling.")
    args = parser.parse_args()
    logger.info(f"Running with arguments: {args}")

    # Convert the arguments to Path objects.
    experiment = args.experiment
    visualize = args.visualize
    metaheuristics = args.metaheuristics
    hh_run_dirs_exp_1 = args.hh_run_dirs_exp_1
    hh_run_dirs_exp_2 = args.hh_run_dirs_exp_2
    nr_of_backbone_switches = args.nr_sw
    base_path = Path(args.base_path)
    coordinator_config_file_path = Path(args.coordinator_config)
    heur_run_config_file_path = Path(args.heur_run_config) if args.heur_run_config else None
    parameter_tuning = args.param_tune
    design_space_plot = args.design_space_plot

    # Check if the base path exists.
    if not base_path.exists():
        raise FileNotFoundError(f"The base path {base_path} does not exist.")

    # Check if any experiments are defined.
    if experiment is None:
        logger.warning("No experiment defined. The program will continue without running any experiment.")

    # Check if the coordinator configuration file exists
    if not coordinator_config_file_path.exists():
        raise FileNotFoundError(f"The coordinator configuration file {coordinator_config_file_path} does not exist.")

    # --- Yappi Profiling Start ---
    logger.info(f"Yappi Profiling Start")
    # Load coordinator_config to get general_files path for yappi output
    profiler_active = False
    profile_dir = None
    if coordinator_config_file_path.exists():
        try:
            coord_conf_for_yappi = Config(coordinator_config_file_path, name="yappi_config_reader")
            general_files_path = coord_conf_for_yappi.tryGet("output_paths", "general_files")
            logger.info(f"General files path: {general_files_path}")

            if general_files_path:
                yappi_output_dir_base = os.path.join(general_files_path, "profiling_yappi")
                # e.g. profile every 5 minutes (300 s)
                # You can adjust interval_seconds as needed
                profile_dir = start_periodic_profiling(interval_seconds=30, output_subdir=yappi_output_dir_base)
                logger.info(f"Yappi profiling started. Snapshots will be written to: {profile_dir}")
                profiler_active = True
            else:
                logger.warn(f"Yappi profiling NOT started: 'output_paths.general_files' not found in coordinator config.")
        except Exception as e:
            logger.warn(f"Yappi profiling NOT started due to error reading coordinator config: {e}")
    else:
        logger.warn(f"Yappi profiling NOT started: Coordinator config file not found at {coordinator_config_file_path}")
    # --- End Yappi Profiling Start ---

    try:
        logger.info(f"Try to run the main function.")
        main(base_path, coordinator_config_file_path, heur_run_config_file_path, experiment, visualize, metaheuristics, hh_run_dirs_exp_1, hh_run_dirs_exp_2, parameter_tuning, design_space_plot, nr_of_backbone_switches)
    finally:
        # --- Yappi Profiling Stop ---
        if profiler_active and yappi.is_running():
            final_snapshot_path = os.path.join(profile_dir, "final_snapshot.pstat")
            try:
                yappi.get_func_stats().save(final_snapshot_path, type="pstat")
                logger.info(f"Yappi: Final profile snapshot saved to {final_snapshot_path}")
            except Exception as e:
                logger.error(f"Yappi: Failed to save final snapshot: {e}")
            yappi.stop()
            logger.info("Yappi profiling stopped.")
        # --- End Yappi Profiling Stop ---
