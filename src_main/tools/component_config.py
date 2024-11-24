import os
from pathlib import Path

from src_main.models import model, modelASML
from src_main.tools.config_reader import Config
import numpy as np

from customhys import metaheuristic as mh

import xml.etree.ElementTree as ET


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


def create_problem_instanceASML(template_xml_file_path, max_wfpm, min_wfpm, min_cost, max_cost, design_point_sim_run, fitness_value_dir):
    template_file = ET.parse(template_xml_file_path)

    nr_of_processor_cores = len(template_file.findall('.//core[@frequency]'))
    nr_of_processor_active = len(template_file.findall('.//core[@frequency]'))

    variable_num = nr_of_processor_cores

    instance_config = dict()

    # Create a formulation of the problem instance.
    instance_config["boundaries"] = {
        f"proc_{i}": [0, 1] for i in range(1, variable_num + 1)
    }
    instance_config["optimal_solution"] = [0.99] * variable_num,
    instance_config["optimal_fitness"] = 0.0

    boundaries = {
        "wfpm": {
            "max": max_wfpm,
            "min": min_wfpm
        },
        "cost": {
            "max": max_cost,
            "min": min_cost       
        }
    }

    min_range = np.array([instance_config['boundaries'][key][0]
                          for key in instance_config['boundaries']])
    max_range = np.array([instance_config['boundaries'][key][1]
                          for key in instance_config['boundaries']])

    problem_instance = modelASML.instanceASML(variable_num,
                    min_range,
                    max_range,
                    instance_config['optimal_solution'],
                    instance_config['optimal_fitness'],
                    'CQN',
                    design_point_sim_run,
                    boundaries,
                    fitness_value_dir)
    
    return problem_instance.get_formatted_problem()


def generate_asml_config(template_xml_file_path, design_point_xml_file_path, configurations):
    tree = ET.parse(template_xml_file_path)
    # cores = tree.findall('.//core[@frequency]')
    # if len(configurations) != len(cores):
    #     raise ValueError('generate_asml_config: configurations and cores in xml are not the same length!')
    
    # print(configurations)
    for configuration in configurations:
        param_name = configuration["param_name"]
        config_pattern = configuration["config_pattern"]
        value = configuration["value"]
        # print(config_pattern)
        elems_pattern = tree.findall(config_pattern[0])
        # print(configuration, len(elems_pattern))
        # print(config_pattern[0])
        if param_name.startswith("processor_freq"):
            elem = elems_pattern[config_pattern[1]] #%len(elems_pattern)
            elem.attrib["cost"] = str(value[1])
            if len(config_pattern) > 2:
                elem.attrib[config_pattern[2]] = value[0]
            else:
                elem.attrib["frequency"] = value[0]
        elif param_name.startswith("num_processor_cores_active"):
            elem = elems_pattern[0]
            elem.attrib["cost"] = str(value[1])
            m = int(value[0])
            # print(value)
            for core in elem.iter("core"):
                if m > 0:
                    core.attrib['active'] = "true"
                else:
                    core.attrib['active'] = "false"
                m -= 1
                # print(m)
            
    
    tree.write(design_point_xml_file_path)


    

def create_problem_instance(nr_of_backbone_switches, max_datarate, min_datarate, max_cost, min_cost, design_point_sim_run, fitness_value_dir):
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
        boundaries,
        fitness_value_dir
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
