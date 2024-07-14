import os
from pathlib import Path

from src_main.models import model
from src_main.tools.config_reader import Config

from customhys import metaheuristic as mh


def determine_heuristic_space(search_operator_space_path):
    # Open the file for reading
    with open(search_operator_space_path, 'r') as file:
        # Read all lines from the file
        lines = file.readlines()

    # Initialize an empty list to store the tuples
    heurstic_space = []

    # Iterate over each line in the file
    for line in lines:
        # Strip any leading/trailing whitespace from the line
        line = line.strip()
        # Use eval to convert the string representation of the tuple into an actual tuple
        tuple_item = eval(line)
        # Append the tuple to the list
        heurstic_space.append(tuple_item)

    # Print the list of tuples
    return heurstic_space


def create_mh(metaheuristic_path, metaheuristic_name, nr_of_agents, nr_of_iterations, prob, base_path, verbose=False):
    full_metaheuristic_path = Path(os.path.join(base_path, metaheuristic_path))

    # Create a configuration object for the metaheuristic.
    heuristic_run_log_path = os.path.join(base_path, f"data/logs/experiment_2/metaheuristics/{metaheuristic_name}")
    os.makedirs(heuristic_run_log_path, exist_ok=True)
    metaheuristic_config = Config(full_metaheuristic_path, Path(heuristic_run_log_path), "experiment_2_metaheuristic_config_manager")

    # Get the metaheuristic configuration.
    metaheuristic_operators = metaheuristic_config.tryGet("metaheuristic")
    formatted_metaheuristic = _format_metaheuristic(metaheuristic_operators)

    # Generate a metaheuristic object for the problem instance.
    met = mh.Metaheuristic(prob, formatted_metaheuristic, num_agents=nr_of_agents, num_iterations=nr_of_iterations)
    met.verbose = verbose
    return met


def _format_metaheuristic(metaheuristic_operators):
    """
    Defines the heuristic operators based on the given heuristic run configuration.

    Args:
        heuristic_run_config (dict): The configuration for the heuristic run.

    Returns:
        list: A list of tuples, where each tuple contains the name, parameters, and strategy
              of a heuristic operator.
    """
    return [
        (
            metaheuristic_operator["name"],
            metaheuristic_operator["params"],
            metaheuristic_operator["strategy"]
        )
        for metaheuristic_operator in metaheuristic_operators
    ]


def create_problem_instance(nr_of_backbone_switches, max_datarate, min_datarate, max_cost, min_cost, design_point_sim_run):
    """
    Create a problem instance for the CUSTOMHys framework.

    Args:
        run (int): The number of runs for the problem instance.
        heur_sim_coordinator (object): The heuristic simulation coordinator object.

    Returns:
        obj: The formatted problem instance.

    """
    subnet_structure = dict()

    nr_of_backbone_cables = nr_of_backbone_switches - 1

    # Create a formulation of the problem instance.
    subnet_structure["boundaries"] = {
        f"cable_{i}": [0, 1] for i in range(1, nr_of_backbone_cables + 1)
    }
    subnet_structure["optimal_solution"] = [0.9] * nr_of_backbone_switches,
    subnet_structure["optimal_fitness"] = 0.0

    # TODO: Initialize model object from here instead of calling the generate_instance method.
    # Create a problem instance from the formulation.
    boundaries = {
        "datarate": {
            "max": max_datarate,
            "min": min_datarate
        },
        "cost": {
            "max": max_cost,
            "min": min_cost
        }
    }
    problem_instance = model.generate_instance(
        nr_of_backbone_switches,
        subnet_structure,
        design_point_sim_run,
        boundaries
    )
    return problem_instance.get_formatted_problem()


def generate_inet_lans_config(template_ini_file_path, design_point_ini_file_path, configurations, nr_of_backbone_switches):
        """
        Write a new INI file based on an existing template file, with updated configurations and number of backbone switches.

        Parameters:
        template_ini_file_path (str): The path to the template INI file.
        design_point_ini_file_path (str): The path to the new INI file to be created.
        configurations (list): A list of dictionaries, where each dictionary contains a configuration pattern and its corresponding value.
        nr_of_backbone_switches (int): The number of backbone switches to be set in the new INI file.

        Returns:
        None
        """
        with open(template_ini_file_path, 'r') as template_file, open(design_point_ini_file_path, 'w') as new_file:
            for line in template_file:
                # Define the number of backbone switches.
                if line.startswith("LargeNet.n"):
                    new_file.write(f"LargeNet.n = {nr_of_backbone_switches}   # number of switches on backbone\n")
                # TODO: Make this cleaner and more generic.
                elif line.endswith("Remaining traffic"):
                    new_file.write(f'LargeNet.llanBB[1..{nr_of_backbone_switches - 1}].*.cli.destAddress = "serverC"')
                else:
                    new_file.write(line)

            new_file.write("\n# Custom parameters\n")

            for configuration in configurations:
                config_pattern = configuration["config_pattern"]
                value = configuration["value"]
                new_file.write(f"{config_pattern} = {value}\n")

        # # Validate the newly created INI file
        # with open(design_point_ini_file_path, 'r') as config_file:
        #     config_data = config_file.read()
        # _validate_inet_lans_config(config_data)


def _validate_inet_lans_config(config_data):
    required_sections = ['General', 'Config LAN1']
    for section in required_sections:
        if section not in config_data:
            raise ValueError(f"Missing required section: {section}")
    # Add more validation logic as needed
