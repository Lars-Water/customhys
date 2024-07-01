import os

from customhys import hyperheuristic as hh
import src_main.models.model as model


def run_experiment(experiment_config, max_datarate, min_datarate, max_cost, min_cost, design_point_sim_run):
    '''
        Experimental run of tuning the parameters of any provided search operators.
    '''
    search_operator_space_paths = experiment_config.tryGet('search_operator_space_paths')

    # Create problem instance.
    run = 6
    prob = create_problem_instance(run, max_datarate, min_datarate, max_cost, min_cost, design_point_sim_run)

    for search_operator_space_path in search_operator_space_paths:
        experiment_name = "experiment_1_" + os.path.splitext(os.path.basename(search_operator_space_path))[0]

        heuristic_space = determine_heuristic_space(search_operator_space_path)

        # Configure Hyperheurstic Object.
        hh_parameters = experiment_config.tryGet('hh_parameters')
        hyp = hh.Hyperheuristic(
            heuristic_space=heuristic_space,
            problem=prob,
            parameters=hh_parameters,
            file_label="INET-LANS_" + experiment_name
        )

        # # Start hyper-heuristic run.
        # best_sol, best_perf, hist_curr, hist_best = hyp.solve()

        # print(f"Best solution: {best_sol}")
        # print(f"Best performance: {best_perf}")
        # print(f"Current history: {hist_curr}")
        # print(f"Best history: {hist_best}")


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


def create_problem_instance(run, max_datarate, min_datarate, max_cost, min_cost, design_point_sim_run):
    subnet_structure = dict()

    nr_of_backbone_cables = run - 1

    # Create a formulation of the problem instance.
    subnet_structure["boundaries"] = {
        f"cable_{i}": [0, 1] for i in range(1, nr_of_backbone_cables + 1)
    }
    subnet_structure["optimal_solution"] = [0.9] * run,
    subnet_structure["optimal_fitness"] = 30

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
        run,
        subnet_structure,
        design_point_sim_run,
        boundaries
    )
    return problem_instance.get_formatted_problem()
