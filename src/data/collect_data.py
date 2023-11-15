import numpy as np
import json
import os

from datetime import datetime
import time


'''
    Collect the data from the metaheuristic run.

    Args:
        mh (Metaheuristic): The metaheuristic object.
        path (str): The path to the data file.
'''
def collect_mh_run(mh, path, run, num_agents, num_iterations):
    
    heuristic_run = dict()

    heuristic_run = {
        "nr_of_components": run,
        "nr_of_agents": num_agents,
        "nr_of_iterations": num_iterations
    }

    # Convert the current time to a Unix timestamp
    unix_time = int(time.mktime(datetime.now().timetuple()))
    file_name = f"{str(unix_time)}.json"

    # Convert the list of arrays into a single numpy array
    fitness_values = np.asarray(mh.historical["fitness"]).tolist()
    configurations = np.array(mh.historical["position"]).tolist()

    # Collect data from the historical data of the MH run.
    heuristic_run["historical_data"] = {
        "best_fitness_after_every_iteration": fitness_values,
        "best_configuration_after_every_iteration": configurations
    }

    # Find the optimal solution from the MH run.
    optimal_fitness = min(fitness_values)
    optimal_configuration = fitness_values.index(optimal_fitness)

    # Collect the optimal solution from the MH run.
    heuristic_run["optimal_found_solution"] = {
        "optimal_found_fitness": optimal_fitness,
        "optimal_found_configuration": configurations[optimal_configuration]
    }

    # TODO: Collect the metadata of the MH run.

    # TODO Write the data to a file.
    write_data(heuristic_run, path, file_name)


'''
    Collect the data from the hyperheuristic run.

    Args:
        hh (Hyperheuristic): The hyperheuristic object.
        path (str): The path to the data file.
        file_name (str): The name of the data file.
'''
def collect_hh_run(hh, path, file_name):
    pass


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


'''
    Transform the positions from the heuristic run.

    Args:
        positions (list): The list of agent positions from the heuristic run.
'''
def transform_positions_to_configurations(positions):
    
    # Assuming mh.historical["position"] is your source of position data
    positions_array = np.array(positions)

    # Apply the transformation to the entire array
    transformed_positions = (positions_array + 1) / 2

    # If you need the result as a list of lists (optional)
    configurations = transformed_positions.tolist()

    return configurations
