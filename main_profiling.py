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

from setuptools import setup, find_packages
from pathlib import Path

sys.path.insert(0, Path(__file__).parent.as_posix())
sys.path.ap
# TODO: Improve and place this function in a separate module.
import shutil
from datetime import datetime

# ------------------------------------------------------------------------------
# Yappi Profiling Functions
# ------------------------------------------------------------------------------
def dump_yappi_stats(interval_seconds, output_dir):
    """
    Every 'interval_seconds' seconds, dump out current Yappi stats to a .pstat file.
    """
    os.makedirs(output_dir, exist_ok=True)

    counter = 0
    while True:
        time.sleep(interval_seconds)

        # Collect the stats object
        stats = yappi.get_func_stats()

        # Build a filename that reflects epoch or a counter
        timestamp = int(time.time())
        fname = os.path.join(output_dir, f"yappi_snapshot_{timestamp}_{counter}.pstat")

        # Save in pstat (compatible with pstats / SnakeViz)
        stats.save(fname, type="pstat")
        
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
         writes out a .pstat file under output_subdir/SLURM_JOBID_TASKID_*.pstat
    """
    # 1) Grab SLURM-assigned identifiers (to disambiguate per-task files)
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "nojob")
    slurm_ntask = os.environ.get("SLURM_NTASKS", "1") # Changed from SLURM_NTASKS to SLURM_NTASKS to match typical SLURM env
    slurm_task_id = os.environ.get("SLURM_PROCID", "task0") # Changed from SLURM_PROCID to SLURM_PROCID to match typical SLURM env
    
    # 2) Derive a directory unique to this task
    #    If not in SLURM, this will be output_subdir/jobnojob_ntasks1_tasktask0
    task_specific_subdir = f"job{slurm_job_id}_ntasks{slurm_ntask}_task{slurm_task_id}"
    this_output_dir = os.path.join(output_subdir, task_specific_subdir)
    os.makedirs(this_output_dir, exist_ok=True)

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
    conf = Config(coordinator_config_file_path, name = "main_conf_logPath")
    log_path = conf.tryGet("output_paths", "log_files")
    if log_path is None:
        log_path = os.path.join(base_path, "data/logs/")

    experiment_1_config_file_path = Path(os.path.join(base_path, "config/experiment_asml_04_2025/experiment_asml_04_2025.json"))
    experiment_1_log_path = os.path.join(log_path, "experiments/experiment_asml_04_2025")

    config_manager_filename = f"experiment_asml_config_manager_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    experiment_1_config = Config(experiment_1_config_file_path, Path(experiment_1_log_path), config_manager_filename)
    # TODO: Change this naming flow, because it goes from main, to coordinator, to data collector.
    nr_of_agents = experiment_1_config.tryGet("hh_parameters", "num_agents")
    run_name = "experiment_asml"
    coordinator_params = (base_path, coordinator_config_file_path, nr_of_agents, run_name)

    # Run Experiment 1.
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
    print("##### Customhys version:" + str(customhys.__version__))
    # Run the requested experiments.
    if experiment == '1':
        experiment_1(base_path, coordinator_config_file_path, nr_of_backbone_switches)
    elif experiment == '2':
        experiment_2(base_path, coordinator_config_file_path)
    elif experiment == 'asml':
        experiment_asml(base_path, coordinator_config_file_path)
    elif experiment == 'all':
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
    args = parser.parse_args()

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
        warnings.warn("No experiment defined. The program will continue without running any experiment.", UserWarning)

    # Check if the coordinator configuration file exists
    if not coordinator_config_file_path.exists():
        raise FileNotFoundError(f"The coordinator configuration file {coordinator_config_file_path} does not exist.")

    # --- Yappi Profiling Start ---
    # Load coordinator_config to get general_files path for yappi output
    profiler_active = False
    profile_dir = None
    if coordinator_config_file_path.exists():
        try:
            coord_conf_for_yappi = Config(coordinator_config_file_path, name="yappi_config_reader")
            general_files_path = coord_conf_for_yappi.tryGet("output_paths", "general_files")

            if general_files_path:
                yappi_output_dir_base = os.path.join(general_files_path, "profiling_yappi")
                # e.g. profile every 5 minutes (300 s)
                # You can adjust interval_seconds as needed
                profile_dir = start_periodic_profiling(interval_seconds=300, output_subdir=yappi_output_dir_base)
                print(f"[INFO] Yappi profiling started. Snapshots will be written to: {profile_dir}")
                profiler_active = True
            else:
                print("[WARN] Yappi profiling NOT started: 'output_paths.general_files' not found in coordinator config.")
        except Exception as e:
            print(f"[WARN] Yappi profiling NOT started due to error reading coordinator config: {e}")
    else:
        print(f"[WARN] Yappi profiling NOT started: Coordinator config file not found at {coordinator_config_file_path}")
    # --- End Yappi Profiling Start ---

    try:
        main(base_path, coordinator_config_file_path, heur_run_config_file_path, experiment, visualize, metaheuristics, hh_run_dirs_exp_1, hh_run_dirs_exp_2, parameter_tuning, design_space_plot, nr_of_backbone_switches)
    finally:
        # --- Yappi Profiling Stop ---
        if profiler_active and yappi.is_running():
            final_snapshot_path = os.path.join(profile_dir, "final_snapshot.pstat")
            try:
                yappi.get_func_stats().save(final_snapshot_path, type="pstat")
                print(f"[INFO] Yappi: Final profile snapshot saved to {final_snapshot_path}")
            except Exception as e:
                print(f"[ERROR] Yappi: Failed to save final snapshot: {e}")
            yappi.stop()
            print("[INFO] Yappi profiling stopped.")
        # --- End Yappi Profiling Stop ---
