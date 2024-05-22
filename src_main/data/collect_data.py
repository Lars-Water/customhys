import numpy as np
import json
import pandas as pd
import os

from datetime import datetime
import time


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


    def store_design_point_metrics(self, latency_df, cost_df, sim_uid):
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
        append_design_points_metric_output_file = os.path.join(self.dir_design_points_metrics_output, "design_point_metrics.csv")

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
