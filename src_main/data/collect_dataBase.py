import math
import numpy as np
import pandas as pd
import json
import os
import re
import glob
from datetime import datetime
import time
from pathlib import Path
from threading import Lock

from src_main.tools.config_reader import Config
from src_main.visualization import visualization
import src_main.tools.component_config as component_config
import src_main.data.sim_configurations as sim_configurations


class DataCollectorBase:
    def __del__(self):
        self._save_to_csv(force=True)

    def __init__(self, 
                 dir_design_points_metrics_output, 
                 heuristic_name=None,
                 index_col = ['SimulationID'], 
                 save_every_x_seconds = 1800, # 30 minutes * 60 seconds
                 save_every_x_entries = 5000,
                 cleanup_after_x_seconds = 3600 # Default to 1 hour
                ):
        """
        Initialize the CollectData object.

        Args:
            dir_design_points_metrics_output (str): The directory path for storing design points metrics output.
            heuristic_name (str, optional): Name of the heuristic, used in filenames.
            index_col (list, optional): Column(s) to use as index for the DataFrame. Defaults to ['SimulationID'].
            save_every_x_seconds (int, optional): Interval in seconds to save interim data.
            save_every_x_entries (int, optional): Interval in number of entries to save interim data.
            cleanup_after_x_seconds (int, optional): Interval in seconds to move data to final CSV.
        """
        self.dir_design_points_metrics_output = dir_design_points_metrics_output
        os.makedirs(self.dir_design_points_metrics_output, exist_ok=True)
        self.heuristic_name = heuristic_name
        self.lock_metricsDF = Lock()
        self.lock_csv = Lock()
        self.locks_raw_backup_csv = {} # Dictionary to hold a lock for each heuristic_name's raw backup
        self.lock_for_managing_raw_backup_locks = Lock() # Lock for managing the above dictionary

        self.index_col = index_col
        # Ensure 'last_touched_timestamp' is part of the columns, not the index
        initial_columns = list(index_col) # Make a mutable copy
        if 'last_touched_timestamp' not in initial_columns:
            initial_columns.append('last_touched_timestamp')
        
        self.metricsDF = pd.DataFrame(columns=initial_columns)
        if index_col and index_col[0] in initial_columns: # Typically 'SimulationID'
            self.metricsDF = self.metricsDF.set_index(index_col[0]) # Assuming single index for simplicity now
            if 'last_touched_timestamp' in self.metricsDF.index.names: # Should not happen with above logic
                 # This case needs more robust handling if 'last_touched_timestamp' can be an index
                 pass

        self.last_save_time = time.time()
        self.last_save_length = 0
        self.save_every_x_seconds = save_every_x_seconds
        self.save_every_x_entries = save_every_x_entries
        self.cleanup_after_x_seconds = cleanup_after_x_seconds
        self._new_data_to_save = False

        self.metrics_defaults = {}


    def _store_design_point_metrics_df(self, heuristic_name, append_df_original): # Renamed parameter
        if not self.heuristic_name and heuristic_name:
            self.heuristic_name = heuristic_name

        if self.heuristic_name and not heuristic_name:
            heuristic_name = self.heuristic_name    

        # --- Start: New Raw Backup Logic --- 
        if heuristic_name: 
            raw_backup_csv_path = os.path.join(
                self.dir_design_points_metrics_output,
                f"design_point_metrics_{heuristic_name}_raw_backup.csv"
            )
            
            # Get or create the specific lock for this heuristic_name's raw backup file
            with self.lock_for_managing_raw_backup_locks:
                if heuristic_name not in self.locks_raw_backup_csv:
                    self.locks_raw_backup_csv[heuristic_name] = Lock()

            with self.locks_raw_backup_csv[heuristic_name]: # Use the specific lock
                append_df_original.to_csv(
                    raw_backup_csv_path,
                    mode='a',
                    header=not os.path.exists(raw_backup_csv_path),
                    index=False 
                )
        # --- End: New Raw Backup Logic ---

        # Now, prepare a copy of the original DataFrame for merging into self.metricsDF
        append_df_for_main_metrics = append_df_original.copy()

        current_time = time.time()
        # Ensure 'last_touched_timestamp' is in append_df_for_main_metrics
        if 'last_touched_timestamp' not in append_df_for_main_metrics.columns:
            append_df_for_main_metrics['last_touched_timestamp'] = current_time
        else: # This case is unlikely if append_df_original comes directly from DataCollectorASML
            append_df_for_main_metrics['last_touched_timestamp'] = append_df_for_main_metrics['last_touched_timestamp'].fillna(current_time)

        # Set index to 'SimulationID' for append_df_for_main_metrics to align with self.metricsDF
        # self.metricsDF is expected to be indexed by 'SimulationID'
        if self.index_col[0] in append_df_for_main_metrics.columns:
            append_df_for_main_metrics = append_df_for_main_metrics.set_index(self.index_col[0])
        elif append_df_for_main_metrics.index.name != self.index_col[0]:
            # This is a fallback or error case if SimulationID was not a column as expected
            # Potentially log a warning or raise an error if SimulationID is missing for indexing
            print(f"Warning: {self.index_col[0]} column not found in append_df for main metrics processing. Index: {append_df_for_main_metrics.index.name}")
            pass # Or raise an error

         
        with self.lock_metricsDF:  
            # Add new rows from append_df_for_main_metrics
            new_rows_mask = ~append_df_for_main_metrics.index.isin(self.metricsDF.index)
            if new_rows_mask.any():
                self.metricsDF = pd.concat([self.metricsDF, append_df_for_main_metrics[new_rows_mask]])
            
            # Update existing rows in self.metricsDF from append_df_for_main_metrics
            existing_rows_mask = append_df_for_main_metrics.index.isin(self.metricsDF.index)
            if existing_rows_mask.any():
                # For update, ensure that the timestamp is also updated if other values are.
                # The 'last_touched_timestamp' is already in append_df_for_main_metrics.
                self.metricsDF.update(append_df_for_main_metrics[existing_rows_mask])
                # If update() doesn't hit NA values correctly for timestamp, explicitly set it:
                for sim_id in append_df_for_main_metrics[existing_rows_mask].index:
                     if pd.isna(self.metricsDF.at[sim_id, 'last_touched_timestamp']):
                          self.metricsDF.at[sim_id, 'last_touched_timestamp'] = current_time
            
            self._new_data_to_save = True
        
        self._save_to_csv()

    def append_caching_status_to_design_point_metrics(self, uids, cached):
        self.metrics_defaults["Cached"] = {
            "default": False,
            "type": 'bool'
        }
        with self.lock_metricsDF:
            current_time = time.time()
            for sim_uid in uids:
                if sim_uid not in self.metricsDF.index:
                    # Add new row if sim_uid doesn't exist, including the timestamp
                    new_data = {'Cached': cached, 'last_touched_timestamp': current_time}
                    new_row = pd.DataFrame([new_data], index=pd.Index([sim_uid], name=self.metricsDF.index.name))
                    self.metricsDF = pd.concat([self.metricsDF, new_row])
                else:
                    # Update existing row
                    self.metricsDF.at[sim_uid, "Cached"] = cached
                    self.metricsDF.at[sim_uid, "last_touched_timestamp"] = current_time
            self._new_data_to_save = True
        
        self._save_to_csv()

    def append_fitness_and_hh_data_to_design_point_metrics(self, uids, fitness_values, search_operator, step, iteration):
        with self.lock_metricsDF:
            current_time = time.time()
            for sim_uid, agent_id in uids.items():
                data_to_update = {
                    "Fitness": fitness_values[agent_id],
                    "SearchOperator": search_operator,
                    "step": step,
                    "iteration": iteration,
                    "last_touched_timestamp": current_time
                }
                if sim_uid not in self.metricsDF.index:
                    new_row = pd.DataFrame([data_to_update], index=pd.Index([sim_uid], name=self.metricsDF.index.name))
                    self.metricsDF = pd.concat([self.metricsDF, new_row])
                else:
                    for col, val in data_to_update.items():
                        self.metricsDF.at[sim_uid, col] = val
            self._new_data_to_save = True
        
        self._save_to_csv()

    def _save_to_csv(self, force=False):
        if not self._new_data_to_save and not force:
            return

        # Define filenames
        base_filename = f"design_point_metrics_{self.heuristic_name}"
        final_csv_path = os.path.join(self.dir_design_points_metrics_output, f"{base_filename}_final.csv")
        interim_csv_path = os.path.join(self.dir_design_points_metrics_output, f"{base_filename}_interim.csv")

        # Save conditions for interim save
        triggered_by_time_interim = (time.time() - self.last_save_time) >= self.save_every_x_seconds
        triggered_by_entries_interim = (len(self.metricsDF.index) - self.last_save_length) >= self.save_every_x_entries
        
        # Always perform cleanup check if saving, or if forced (especially for __del__)
        perform_save_action = force or triggered_by_time_interim or triggered_by_entries_interim

        if self.heuristic_name is not None and self.heuristic_name != "" and perform_save_action:
            with self.lock_csv:
                df_to_finalize = pd.DataFrame()
                
                with self.lock_metricsDF:
                    # Apply defaults before any operations
                    self._set_metricsDF_defaults(acquire_lock=False) # Modifies self.metricsDF in place

                    if 'last_touched_timestamp' in self.metricsDF.columns:
                        current_time_for_cleanup = time.time()
                        # Identify rows to move to final CSV
                        # Ensure last_touched_timestamp is numeric for comparison
                        self.metricsDF['last_touched_timestamp'] = pd.to_numeric(self.metricsDF['last_touched_timestamp'], errors='coerce')
                        
                        finalize_mask = (current_time_for_cleanup - self.metricsDF['last_touched_timestamp']) > self.cleanup_after_x_seconds
                        df_to_finalize = self.metricsDF[finalize_mask].copy() # Explicit copy
                        
                        # Remove finalized rows from in-memory DataFrame
                        if not df_to_finalize.empty:
                            self.metricsDF = self.metricsDF[~finalize_mask].copy() # Explicit copy

                    df_interim_to_save = self.metricsDF.copy() # What remains is interim

                # Save finalized data (append mode)
                if not df_to_finalize.empty:
                    df_to_finalize.to_csv(final_csv_path, mode='a', header=not os.path.exists(final_csv_path), index=True)
                    print(f"Moved {len(df_to_finalize.index)} rows to {final_csv_path}")

                # Save interim data (overwrite mode)
                df_interim_to_save.to_csv(interim_csv_path, index=True)
                print(f"Saved {len(df_interim_to_save.index)} rows to {interim_csv_path}")

                self.last_save_time = time.time()
                self.last_save_length = len(df_interim_to_save.index) # Length of what remains as interim
                self._new_data_to_save = False
                
                # Original print logic for why interim save was triggered (can be adapted)
                if force and not (triggered_by_time_interim or triggered_by_entries_interim): # Avoid double printing if force coincided
                     print(f"Interim save forced. Finalized: {len(df_to_finalize.index)}, Interim: {len(df_interim_to_save.index)}")
                elif triggered_by_time_interim:
                    print(f"Interim save due to time. Finalized: {len(df_to_finalize.index)}, Interim: {len(df_interim_to_save.index)}")
                elif triggered_by_entries_interim:
                    print(f"Interim save due to entries. Finalized: {len(df_to_finalize.index)}, Interim: {len(df_interim_to_save.index)}")


        elif force: # Handle force save, attempting to use standard filenames if possible
            with self.lock_csv:
                final_csv_path = ""
                interim_csv_path = ""
                log_prefix = "Forced save: "

                if self.heuristic_name:
                    base_filename = f"design_point_metrics_{self.heuristic_name}"
                    final_csv_path = os.path.join(self.dir_design_points_metrics_output, f"{base_filename}_final.csv")
                    interim_csv_path = os.path.join(self.dir_design_points_metrics_output, f"{base_filename}_interim.csv")
                    log_prefix = f"Forced save for heuristic '{self.heuristic_name}': "
                else:
                    # Fallback to timestamped names if heuristic_name is not available
                    timestamp_str = int(time.time())
                    final_csv_path = os.path.join(self.dir_design_points_metrics_output, f"design_point_metrics_forced_final_{timestamp_str}.csv")
                    interim_csv_path = os.path.join(self.dir_design_points_metrics_output, f"design_point_metrics_forced_interim_{timestamp_str}.csv")
                    log_prefix = f"Forced save (no heuristic_name, using timestamp {timestamp_str}): "
                
                # Ensure output directory exists (especially important for fallback names)
                os.makedirs(self.dir_design_points_metrics_output, exist_ok=True)

                with self.lock_metricsDF:
                    self._set_metricsDF_defaults(acquire_lock=False) # Lock already held by caller
                    
                    df_to_finalize_on_force = pd.DataFrame()
                    if 'last_touched_timestamp' in self.metricsDF.columns:
                        current_time_for_cleanup = time.time()
                        # Ensure last_touched_timestamp is numeric for comparison
                        self.metricsDF['last_touched_timestamp'] = pd.to_numeric(self.metricsDF['last_touched_timestamp'], errors='coerce')
                        
                        finalize_mask = (current_time_for_cleanup - self.metricsDF['last_touched_timestamp']) > self.cleanup_after_x_seconds
                        df_to_finalize_on_force = self.metricsDF[finalize_mask].copy()
                        if not df_to_finalize_on_force.empty:
                            self.metricsDF = self.metricsDF[~finalize_mask].copy()

                    df_interim_on_force = self.metricsDF.copy() # What remains is interim

                # Save finalized data (append mode)
                if not df_to_finalize_on_force.empty:
                    df_to_finalize_on_force.to_csv(final_csv_path, mode='a', header=not os.path.exists(final_csv_path), index=True)
                    print(f"{log_prefix}Moved {len(df_to_finalize_on_force.index)} rows to {final_csv_path}")

                # Save interim data (overwrite mode)
                if not df_interim_on_force.empty or self._new_data_to_save: # only save if there's something or new data flag was set
                    df_interim_on_force.to_csv(interim_csv_path, index=True)
                    print(f"{log_prefix}Saved {len(df_interim_on_force.index)} rows to {interim_csv_path}")
                
                self._new_data_to_save = False

    def _set_metricsDF_defaults(self, acquire_lock=True):
        # This method manipulates self.metricsDF. 
        # If acquire_lock is True, it will manage the lock itself.
        # If acquire_lock is False (default), it assumes the caller has already acquired the lock.
        
        if acquire_lock:
            with self.lock_metricsDF:
                self._INTERNAL_apply_defaults_to_df() # Call a helper to do the actual work
        else:
            # Assumes lock is already held by the caller
            self._INTERNAL_apply_defaults_to_df()

    def _INTERNAL_apply_defaults_to_df(self):
        # Helper method containing the original logic of _set_metricsDF_defaults
        # This is always called when self.lock_metricsDF is held (either by this method or by caller)
        for key, val in self.metrics_defaults.items():
            if key in self.metricsDF.columns:
                with pd.option_context('future.no_silent_downcasting', True):
                    if type(val) == dict and "type" in val.keys() and "default" in val.keys():
                        self.metricsDF[key] = self.metricsDF[key].fillna(val["default"]).astype(val["type"])
                    if type(val) == dict and "default" in val.keys(): # Check again in case first condition was false but this is true
                        self.metricsDF[key] = self.metricsDF[key].fillna(val["default"])
                    else:
                        self.metricsDF[key] = self.metricsDF[key].fillna(val)

            
        
def _natural_key(file_name):
    """
    Generate a key for natural sorting of file names.

    This function splits the input file name into a list of integers and non-digit parts,
    which can be used to sort file names in a human-friendly order (e.g., 'file2' comes
    before 'file10').

    Args:
        file_name (str): The name of the file to be split into parts.

    Returns:
        list: A list containing integers and strings, representing the split parts of the file name.
    """
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', file_name)]


def read_json_files(directory_path):
    """
    Reads every JSON file in the specified directory and returns their contents as a list of dictionaries.

    Parameters:
        directory_path (str): The path to the directory containing JSON files.

    Returns:
        list: A list of tuples, each containing the file name and the dictionary representing the contents of a JSON file.
    """
    json_contents = []

    # Check if the specified path is a directory
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"The path '{directory_path}' is not a valid directory.")

    # List all files in the directory and sort using the natural_key
    file_names = sorted(os.listdir(directory_path), key=_natural_key)

    for file_name in file_names:
        # Check if the file is a JSON file
        if file_name.endswith('.json'):
            file_path = os.path.join(directory_path, file_name)

            # Open and read the JSON file
            with open(file_path, 'r', encoding='utf-8') as json_file:
                try:
                    data = json.load(json_file)
                    json_contents.append((file_name, data))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file '{file_name}': {e}")

    return json_contents


def quick_and_dirty_collect_hh_fitness_for_every_step(hh_run_path):
    # Read the JSON files from the specified directory.
    hh_steps = read_json_files(hh_run_path)

    # Process the fitness values for every hh replica.
    fitness_values_list = []
    for _, data in hh_steps:
        if 'details' in data and 'historical' in data['details'] and isinstance(data['details']['historical'], list) and 'fitness' in data['details']['historical'][0]:
            fitness_values_list.append(data['details']['historical'][0]['fitness'])
    merged_fitness_values = [fitness for fitness_values in fitness_values_list for fitness in fitness_values]

    return merged_fitness_values


def _find_extreme_avg_files(json_data):
    """
    Finds the JSON files with the highest and lowest 'Avg' values in the 'statistics' key.

    Parameters:
        json_data (list): A list of tuples, each containing the file name and the dictionary representing the contents of a JSON file.

    Returns:
        tuple: The file name with the highest 'Avg' value and the file name with the lowest 'Avg' value.
    """
    max_avg = None
    min_avg = None
    max_avg_hh_step = None
    min_avg_hh_step = None
    max_avg_file_name = None
    min_avg_file_name = None

    # Iterate through the list of JSON data
    for file_name, data in json_data:
        # Check if 'statistics' and 'Avg' key exist in the JSON
        if 'details' in data and 'statistics' in data['details'] and 'Avg' in data['details']['statistics']:
            avg_value = data['details']['statistics']['Avg']

            if max_avg is None:
                max_avg = avg_value
                max_avg_hh_step = data
                max_avg_file_name = file_name
            # Update the file with the highest Avg value
            elif avg_value > max_avg:
                max_avg = avg_value
                max_avg_hh_step = data
                max_avg_file_name = file_name
            if min_avg is None:
                min_avg = avg_value
                min_avg_hh_step = data
                min_avg_file_name = file_name
            # Update the file with the lowest Avg value
            elif avg_value < min_avg:
                min_avg = avg_value
                min_avg_hh_step = data
                min_avg_file_name = file_name

    print(f"File with worst 'Avg' fitness: {max_avg_file_name}")
    print(f"File with best 'Avg' fitness: {min_avg_file_name}")
    return max_avg_hh_step, min_avg_hh_step


def _process_hh_step_positions(centre_boundaries, span_boundaries, agent_positions, nr_of_cable_types, rescale=True):
    if rescale:
        return [math.ceil((centre + position * (span / 2)) * nr_of_cable_types) if math.ceil((centre + position * (span / nr_of_cable_types)) * nr_of_cable_types) != 0 else 1 for centre, span, position in zip(centre_boundaries, span_boundaries, agent_positions)]
    else:
        return [math.ceil(position * nr_of_cable_types) if math.ceil(position * nr_of_cable_types) != 0 else 1 for position in agent_positions]


def quick_and_dirty_save_hh_positions_to_xlsx(hh_run_path, nr_of_backbone_switches, nr_of_cable_types, rescale=False):

    # Define rescaling parameter values.
    centre_boundaries = [0.5] * (nr_of_backbone_switches-1)
    span_boundaries = [1.0] * (nr_of_backbone_switches-1)

    # Read the JSON files from the specified directory.
    hh_steps = read_json_files(hh_run_path)

    # Determine the JSON files for the steps with the highest and lowest average fitness values.
    max_avg_hh_step, min_avg_hh_step = _find_extreme_avg_files(hh_steps)
    if max_avg_hh_step is None or min_avg_hh_step is None:
        raise ValueError("No JSON files with 'statistics' and 'Avg' keys found.")
    positions = {
        "worst_solution": max_avg_hh_step['details']['positions'],
        "best_solution": min_avg_hh_step['details']['positions']
    }

    # Process the optimal cable configuration values for every hh replica.
    optimal_configurations_list = []
    for category, replicas_positions in positions.items():
        for replica_positions in replicas_positions:
            processed_positions = _process_hh_step_positions(centre_boundaries, span_boundaries, replica_positions, nr_of_cable_types, rescale)
            optimal_configurations_list.append([category] + processed_positions)

    sim_configurations.write_processed_design_points_to_xlsx(hh_run_path, optimal_configurations_list, nr_of_backbone_switches)


'''
    Collect the data from the metaheuristic run.

    Args:
        mh (Metaheuristic): The metaheuristic object.
        path (str): The path to the data file.
'''
def collect_mh_run(mh, path, nr_of_components, num_agents, num_iterations, heuristic_run_meta_data):

    heuristic_run = dict()

    heuristic_run = {
        "nr_of_components": nr_of_components,
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


def save_hh_run_meta_data(path, best_sol, best_perf, hist_curr, hist_best, hh_run_meta_data):
    # Define the metadata from the mh run.
    metadata = {
        "total_execution_time": hh_run_meta_data["total_execution_time"],
        "heuristic_execution_time": hh_run_meta_data["heuristic_execution_time"],
        "coordinator_execution_time": hh_run_meta_data["coordinator_execution_time"],
        "simulation_execution_time": hh_run_meta_data["simulation_execution_time"],
        "best_sol": best_sol,
        "best_perf": best_perf,
        "hist_curr": hist_curr,
        "hist_best": hist_best
    }

    # Define the filename and write remaining data to JSON file.
    json_file_name = "general.json"
    write_data(metadata, path, json_file_name)


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
    # Check if the data is an ndarray and convert to list if necessary
    if isinstance(data, np.ndarray):
        data = data.tolist()

    # If the data is a dictionary, recursively convert any ndarrays in its values
    elif isinstance(data, dict):
        data = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in data.items()}

    # Similarly handle lists of ndarrays
    elif isinstance(data, list):
        data = [(item.tolist() if isinstance(item, np.ndarray) else item) for item in data]

    # Save data structure.
    os.makedirs(path, exist_ok=True)
    with open( os.path.join(path, file_name) , "w") as file:
        json.dump(data, file, indent=4)


def get_raw_run_paths(directory_path):
    return {folder_name: os.path.join(directory_path, folder_name) for folder_name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder_name)) and folder_name.isdigit()}


def get_processed_folder_ids(directory_path):
    return {folder_name for folder_name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder_name)) and folder_name.isdigit()}


def vizualize_mh_runs(nr_of_backbone_switches):
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
            component_config.generate_inet_lans_config(template_ini_file_path, design_point_ini_file_path, best_solution_configurations, nr_of_backbone_switches)

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
    simulation_execution_time = heur_sim_coordinator.simulation_execution_time
    coordinator_execution_time = heur_sim_coordinator.coordinator_and_simulation_execution_time - heur_sim_coordinator.simulation_execution_time
    heuristic_execution_time = total_execution_time - heur_sim_coordinator.coordinator_and_simulation_execution_time

    return {
        "total_execution_time": total_execution_time,
        "heuristic_execution_time": heuristic_execution_time,
        "coordinator_execution_time": coordinator_execution_time,
        "simulation_execution_time": simulation_execution_time
    }
