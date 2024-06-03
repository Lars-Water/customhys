import numpy as np
import pandas as pd
import json
import os
import glob
from datetime import datetime
import time
from pathlib import Path

from src_main.tools.config_reader import Config
from src_main.visualization import visualization


class DataCollector:

    def __init__(self, weight_latency, weight_cost, dir_design_points_metrics_output):
        """
        Initialize the CollectData object.

        Args:
            weight_latency (float): The weight for the latency metric.
            weight_cost (float): The weight for the cost metric.
            dir_design_points_metrics_output (str): The directory path for storing design points metrics output.

        Returns:
            None
        """
        self.weight_latency = weight_latency
        self.weight_cost = weight_cost
        self.dir_design_points_metrics_output = dir_design_points_metrics_output


    def store_design_point_metrics(self, latency_df, cost_df, sim_uid, heuristic_name):
        """
        Stores the design point metrics in a CSV file.

        Args:
            latency_df (pandas.DataFrame): DataFrame containing latency values.
            cost_df (pandas.DataFrame): DataFrame containing cost values.
            sim_uid (str): Unique identifier for the simulation.

        Returns:
            None
        """
        # Define the file output path.
        os.makedirs(self.dir_design_points_metrics_output, exist_ok=True)
        append_design_points_metric_output_file = os.path.join(self.dir_design_points_metrics_output, f"design_point_metrics_{heuristic_name}.csv")

        # Determine weighted metric values.
        adjusted_latency = latency_df['mean'].mean() * self.weight_latency
        adjusted_network_cost = cost_df['value'].astype(float).sum() * self.weight_cost

        # Append the adjusted values to the design points metrics storage.
        append_df = pd.DataFrame([[sim_uid, adjusted_latency, adjusted_network_cost]],
                                columns=['SimulationID', 'AdjustedLatency', 'AdjustedNetworkCost'])
        append_df.to_csv(append_design_points_metric_output_file, mode='a', header=not os.path.exists(append_design_points_metric_output_file), index=False)


'''
    Collect the data from the metaheuristic run.

    Args:
        mh (Metaheuristic): The metaheuristic object.
        path (str): The path to the data file.
'''
def collect_mh_run(mh, path, run, num_agents, num_iterations, heuristic_run_meta_data):

    heuristic_run = dict()

    heuristic_run = {
        "nr_of_components": run,
        "nr_of_agents": num_agents,
        "nr_of_iterations": num_iterations
    }

    # Convert the list of arrays into a single numpy array
    fitness_values = np.asarray(mh.historical["fitness"])
    configurations = np.asarray([mh.pop.rescale_back(position) for position in mh.historical["position"]])

    unix_time = int(time.mktime(datetime.now().timetuple()))

    # Create a folder with the name of unix time
    folder_name = str(unix_time)
    folder_path = os.path.join(path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Save the historical data to a CSV file in the folder
    csv_file_name = save_heur_iterations_data(fitness_values, configurations, unix_time, folder_path)

    # Save the remaining data to a JSON file in the folder
    save_config_and_meta_data(fitness_values.tolist(), configurations.tolist(), heuristic_run_meta_data, heuristic_run, unix_time, csv_file_name, folder_path)


'''
    Collect the data from the hyperheuristic run.

    Args:
        hh (Hyperheuristic): The hyperheuristic object.
        path (str): The path to the data file.
        file_name (str): The name of the data file.
'''
def collect_hh_run(hh, path, file_name):
    pass


def save_heur_iterations_data(fitness_values, configurations, unix_time, path):
    # Reshape fitness_values to make it a two-dimensional array with a single column
    fitness_values_reshaped = fitness_values.reshape(-1, 1)

    # Concatenate fitness_values and configurations along axis 1 (columns)
    combined_data = np.concatenate([fitness_values_reshaped, configurations], axis=1)

    # Convert the combined NumPy array to a Pandas DataFrame
    df = pd.DataFrame(combined_data)

    # Define column names (optional, for clarity)
    column_names = ['best_fitness_value_untill_current_iteration'] + [f'configuration_point_{i+1}' for i in range(configurations.shape[1])]
    df.columns = column_names

    # Save the DataFrame to a CSV file
    csv_file_name = "convergence.csv"
    df.to_csv(os.path.join(path, csv_file_name), index=False)

    return csv_file_name


def save_config_and_meta_data(fitness_values, configurations, heuristic_run_meta_data, heuristic_run, unix_time, csv_file_name, path):
    # Determine the optimal solution from the MH run.
    optimal_fitness = min(fitness_values)
    optimal_configuration = fitness_values.index(optimal_fitness)
    optimal_solution = {
        "optimal_found_fitness": optimal_fitness,
        "optimal_found_configuration": configurations[optimal_configuration]
    }

    # Define the metadata from the mh run.
    metadata = {
        "total_execution_time": heuristic_run_meta_data["total_execution_time"],
        "heuristic_execution_time": heuristic_run_meta_data["heuristic_execution_time"],
        "coordinator_execution_time": heuristic_run_meta_data["coordinator_execution_time"],
        "simulation_execution_time": heuristic_run_meta_data["simulation_execution_time"]
    }

    # Define the filename and write remaining data to JSON file.
    json_file_name = "general.json"
    remaining_data = {
        "nr_of_components": heuristic_run["nr_of_components"],
        "nr_of_agents": heuristic_run["nr_of_agents"],
        "nr_of_iterations": heuristic_run["nr_of_iterations"],
        "historical_data_file": csv_file_name,
        "optimal_found_solution": optimal_solution,
        "metadata": metadata
    }
    write_data(remaining_data, path, json_file_name)


'''
    Write the data to a file.

    Args:
        data (dict): The data to be written to a file.
        path (str): The path to the data file.
'''
def write_data(data, path, file_name):
    os.makedirs(path, exist_ok=True)
    with open( os.path.join(path, file_name) , "w") as file:
        json.dump(data, file, indent=4)


def get_raw_run_paths(directory_path):
    return {folder_name: os.path.join(directory_path, folder_name) for folder_name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder_name)) and folder_name.isdigit()}


def get_processed_folder_ids(directory_path):
    return {folder_name for folder_name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder_name)) and folder_name.isdigit()}


def vizualize_mh_runs(ini_generation_func, nr_of_backbone_switches):
    # Compare folder ids in raw and processed to vizualize absent processed folders.
    mh_raw_results_path = os.path.join(os.getcwd(), "data/raw/results/metaheuristic")
    raw_mh_categories = os.listdir(mh_raw_results_path)
    raw_run_paths = {run_id: os.path.join(mh_raw_results_path, mh_category, run_id) for mh_category in raw_mh_categories for run_id in get_raw_run_paths(os.path.join(mh_raw_results_path, mh_category))}

    mh_processed_results_path = os.path.join(os.getcwd(), "data/processed/results/metaheuristic")
    processed_mh_categories = os.listdir(mh_processed_results_path)
    processed_run_ids = {run_id for mh_category in processed_mh_categories for run_id in get_processed_folder_ids(os.path.join(mh_processed_results_path, mh_category))}

    for run_id, run_path in raw_run_paths.items():
        if run_id not in processed_run_ids:
            try:
                convergence_data_path = os.path.join(run_path, "convergence.csv")
                general_heuristic_run_info_path = glob.glob(os.path.join(run_path, "*.json"))[0]
                design_points_path = os.path.join(run_path, "design_point_metrics.csv")
            except IndexError:
                raise FileNotFoundError(f"Could not find the best fitness after every iteration or general heuristic run info files in {run_path}")

            processed_run_path = run_path.replace("raw/", "processed/")
            os.makedirs(processed_run_path, exist_ok=True)

            # TODO: Remove hardcoded parameter values.
            # Save the best configuration to an ini file.
            general_heuristic_run_info = Config(Path(general_heuristic_run_info_path), Path(processed_run_path), "general_heuristic_run_info")
            best_configuration_values = general_heuristic_run_info.tryGet("optimal_found_solution", "optimal_found_configuration")
            template_ini_file_path = os.path.join("/home/larry/hyper-heuristic-dse-2.0/src_main/external/simulation_model/sims/dummy_sim_lans", "largeNet.ini")
            design_point_ini_file_path = os.path.join(processed_run_path, "best_configuration.ini")
            best_solution_configurations = generate_best_configuration_values(np.array(best_configuration_values))
            ini_generation_func(template_ini_file_path, design_point_ini_file_path, best_solution_configurations, nr_of_backbone_switches)

            # Vizualize best configuration.
            param_values, counts = determine_cable_occurrences(best_configuration_values)
            output_path_best_solution = os.path.join(processed_run_path, "best_configuration.jpg")
            visualization.plot_cable_occurrences_histogram(param_values, counts, output_path_best_solution)

            # Vizualize convergence.
            output_path_convergence = os.path.join(processed_run_path, "convergence.jpg")
            visualization.main_plot_convergence_csv(convergence_data_path, output_path_convergence)

            # Vizualize Pareto Front.
            if os.path.exists(design_points_path):
                output_path_design_space = os.path.join(processed_run_path, "pareto_front.jpg")
                if "/ga" in run_path:
                    visualization.main_plot_design_space(design_points_path, output_path_design_space, 2)
                else:
                    visualization.main_plot_design_space(design_points_path, output_path_design_space)


# TODO: Refactor such that it is more suited for the framework.
def generate_best_configuration_values(best_configuration_values):
    config_pattern = "**.switchBB[__switch_index].ethg$o[__gate_index].channel.display-string"
    param_values = [
        "ls=red,3,s;",
        "ls=blue,3,s;",
        "ls=green,3,s;",
        "ls=yellow,3,s;"
    ]

    # Determine the param value index by segment indexing the config values from CUSTOMHys.
    num_segments = len(param_values)
    value_indices = np.floor(best_configuration_values * num_segments).astype(int)
    value_indices[best_configuration_values == 1.0] = num_segments - 1

    configurations = []

    # TODO: Optimize this operations as it now parses and has duplicate code.
    # Create a configuration for each parameter.
    for switch_idx, value_idx in enumerate(value_indices):
        selected_param_value = param_values[value_idx]

        # Conditional handling for the switch gates where the first switch is handled differently.
        first_switch_gate_index = "1" if switch_idx > 0 else "0"
        second_switch_gate_index = "0"

        # Replace placeholders in the configuration pattern are replaced with the corresponding indices.
        current_param_pattern = config_pattern.replace("__switch_index", str(switch_idx))
        current_param_pattern = current_param_pattern.replace("__gate_index", first_switch_gate_index)

        # Create a configuration for the current parameter.
        configurations.append({
            "config_pattern": current_param_pattern,
            "value": selected_param_value
        })

        # Replace placeholders in the configuration pattern are replaced with the corresponding indices.
        current_param_pattern = config_pattern.replace("__switch_index", str(switch_idx+1))
        current_param_pattern = current_param_pattern.replace("__gate_index", second_switch_gate_index)

        configurations.append({
            "config_pattern": current_param_pattern,
            "value": selected_param_value
        })

    return configurations


# TODO: Refactor such that it is more suited for the framework.
# TODO: Remove hardcoded parts.
def determine_cable_occurrences(best_configuration_values):
    formatted_best_configuration_values = np.array(best_configuration_values)

    param_values = [
        "10Mbps",
        "100Mbps",
        "1Gbps",
        "10Gbps"
    ]
    num_segments = len(param_values)
    value_indices = np.floor(formatted_best_configuration_values * num_segments).astype(int)
    value_indices[best_configuration_values == 1.0] = num_segments - 1

    # Get unique values and their counts
    _, counts = np.unique(value_indices, return_counts=True)

    return param_values, counts
