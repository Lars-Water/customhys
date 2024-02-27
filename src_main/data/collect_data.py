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
def collect_mh_run(mh, path, run, num_agents, num_iterations, heuristic_run_meta_data):
    
    heuristic_run = dict()

    heuristic_run = {
        "nr_of_components": run,
        "nr_of_agents": num_agents,
        "nr_of_iterations": num_iterations
    }

    # Convert the list of arrays into a single numpy array
    fitness_values = np.asarray(mh.historical["fitness"]).tolist()
    configurations = np.asarray([mh.pop.rescale_back(position) for position in mh.historical["position"]]).tolist()

    # Collect data from the historical data of the MH run.
    heuristic_run["historical_data"] = {
        "best_fitness_after_every_iteration": fitness_values,
        "best_configuration_after_every_iteration": configurations
    }

    # Determine the optimal solution from the MH run.
    optimal_fitness = min(fitness_values)
    optimal_configuration = fitness_values.index(optimal_fitness)
    heuristic_run["optimal_found_solution"] = {
        "optimal_found_fitness": optimal_fitness,
        "optimal_found_configuration": configurations[optimal_configuration]
    }

    # Define the metadata from the mh run.
    heuristic_run["metadata"] = {
        "total_execution_time": heuristic_run_meta_data["total_execution_time"],
        "heuristic_execution_time": heuristic_run_meta_data["heuristic_execution_time"],
        "coordinator_execution_time": heuristic_run_meta_data["coordinator_execution_time"],
        "simulation_execution_time": heuristic_run_meta_data["simulation_execution_time"]
    }

    # Define the filename and write to file.
    unix_time = int(time.mktime(datetime.now().timetuple()))
    file_name = f"{str(unix_time)}.json"
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
