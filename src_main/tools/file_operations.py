from optparse import Values
import os
import shutil
import glob
from itertools import product
from typing import List, Dict, Any


def remove_sim_instance_folders(data_path, uids):
    """
    Remove simulation instance folders.

    This function takes a data path and a list of unique identifiers (uids) as input.
    It iterates over the uids and removes the corresponding simulation instance folders
    from the data path. If a simulation instance folder does not exist, it prints a message.

    Parameters:
    - data_path (str): The path to the data directory.
    - uids (list): A list of unique identifiers (uids) representing the simulation instance folders.

    Returns:
    None
    """
    for uid in uids:
        logs_sim_instance_path = os.path.join(data_path, "logs", uid)
        results_sim_instance_path = os.path.join(data_path, "results", uid)
        runtime_sim_instance_path = os.path.join(data_path, "runtime", uid)
        if os.path.exists(logs_sim_instance_path):
            shutil.rmtree(logs_sim_instance_path)
        else:
            print(f"Simulation instance folder {logs_sim_instance_path} does not exist.")
        if os.path.exists(results_sim_instance_path):
            shutil.rmtree(results_sim_instance_path)
        else:
            print(f"Simulation instance folder {results_sim_instance_path} does not exist.")
        if os.path.exists(runtime_sim_instance_path):
            shutil.rmtree(runtime_sim_instance_path)
        else:
            print(f"Simulation instance folder {runtime_sim_instance_path} does not exist.")


def remove_design_point_configurations_dummy_path(sim_dummy_directory, pattern="custom_dummy_*"):
    if sim_dummy_directory is not None and os.path.exists(sim_dummy_directory):
        simulation_run_dir_pattern = os.path.join(sim_dummy_directory, pattern)
        matching_sim_dummy_dir = [path for path in glob.glob(simulation_run_dir_pattern) if os.path.isdir(path)]
        for directory in matching_sim_dummy_dir:
            if os.path.exists(directory):
                shutil.rmtree(directory)




def remove_design_point_configurations_sims_path(sims_path):
    if os.path.exists(sims_path):
        shutil.rmtree(sims_path)
    os.makedirs(sims_path)


def generate_heuristic_space(search_operator_name: str, experiment_folder: str, config: Dict[str, Any]):
    keys = config.keys()
    values = config.values()
    heuristic_configurations = product(*values)

    search_operator_space_path = os.path.join(os.getcwd(), "config", experiment_folder, f"search_operator_space/{search_operator_name}.txt")
    with open(search_operator_space_path, 'w') as f:
        for configuration in heuristic_configurations:
            *search_operator_values, selector = configuration
            f.write(f"('{search_operator_name}', {dict(zip(keys, search_operator_values))}, '{selector}')\n")
