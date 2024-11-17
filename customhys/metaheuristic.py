# -*- coding: utf-8 -*-
"""
This module contains the Metaheuristic class.

Created on Thu Sep 26 16:56:01 2019

@author: Jorge Mario Cruz-Duarte (jcrvz.github.io), e-mail: jorge.cruz@tec.mx
"""

import numpy as np
from . import operators as Operators
from .population import Population

import os
import pandas as pd

__all__ = ['Metaheuristic', 'Population', 'Operators']
__operators__ = Operators.__all__
__selectors__ = ['greedy', 'probabilistic', 'metropolis', 'all', 'none']


class Metaheuristic:
    """
        This is the Metaheuristic class, each object corresponds to a metaheuristic implemented with a sequence of
        search operators from op, and it is based on a population from Population.
    """
    def __init__(self, problem, search_operators=None, num_agents: int = 30, num_iterations: int = 100,
                 initial_scheme: str = 'random', verbose: bool = False, finalised_positions_previous_step = None, file_name_fitness_values="fitness_values.json", pass_finalised_positions = False):
        """
        Create a population-based metaheuristic by employing different simple search operators.

        :param dict problem:
            This is a dictionary containing the 'function' that maps a 1-by-D array of real values to a real value,
            'is_constrained' flag that indicates the solution is inside the search space, and the 'boundaries' (a tuple
            with two lists of size D). These two lists correspond to the lower and upper limits of domain, such as:
            ``boundaries = (lower_boundaries, upper_boundaries)``

            **Note:** Dimensions (D) of search domain are read from these boundaries. The problem can be obtained from
            the ``benchmark_func`` module.
        :param list search_operators:
            A list of available search operators. These operators must correspond to those available in the
            ``operators`` module. This parameter is mandatory for mataheuristic implementations, for using parts of this
            class, these can be provided as a list of ``operators``.
        :param int num_agents: Optional.
            Number of agents or population size. The default is 30.
        :param int num_iterations: Optional.
            Number of iterations or generations that the metaheuristic is going to perform. The default is 100.

        :param list finalised_positions_previous_step (optional):
            NOTE: CUSTOM CHANGE BY LARS - The finalised positions of the previous step in the HH process. This is used to pass the finalised positions of the previous step to the next step for initialising the positions of the agents.
        :param bool pass_finalised_positions:
            NOTE: CUSTOM CHANGE BY LARS - Flag that determines whether the finalised positions of the previous step should be used for agent initialization in the current step.
        :return: None.
        """
        # Read the problem function
        self.finalisation_conditions = None
        self._problem_function = problem['function']

        # NOTE: CUSTOM BY LARS - PASSING THE PREVIOUS STEP FINALISED AGENT POSITIONS IF PROVIDED
        # Create population
        self.pop = Population(problem['boundaries'], num_agents, problem['is_constrained'], finalised_positions_previous_step=finalised_positions_previous_step)

        # Check and read the search_operators
        if search_operators:
            if not isinstance(search_operators, list):
                search_operators = [search_operators]
            self.perturbators, self.selectors = Operators.process_operators(search_operators)

        # Define the maximum number of iterations
        self.num_iterations = num_iterations

        # Read the number of dimensions
        self.num_dimensions = self.pop.num_dimensions

        # Read the number of agents
        self.num_agents = num_agents

        # Initialise historical variables
        self.historical = dict()

        # Set additional variables
        self.verbose = verbose

        # Set the initial scheme
        self.initial_scheme = initial_scheme

        # Set file_name_fitness_values
        self.file_name_fitness_values = file_name_fitness_values

        # NOTE: CUSTOM CHANGE BY LARS - Set the flag for passing finalised positions.
        self.pass_finalised_positions = pass_finalised_positions


    def apply_initialiser(self, file_label=None):
        """
        :parameter str file_label: Optional
            NOTE: CUSTOM CHANGE BY LARS - File label to use for saving design point data.
        """
        # Set initial iteration
        self.pop.iteration = 0

        # TODO: Make a function in pop that initialized pop. positions conditionally based on flag.
        # Initialise the population
        self.pop.initialise_positions(self.initial_scheme, file_label)  # Default: random

        # Evaluate fitness values
        self.pop.evaluate_fitness(self._problem_function, self.file_name_fitness_values)

        # Update population, particular, and global
        self.pop.update_positions('population', 'all')  # Default
        self.pop.update_positions('particular', 'all')
        self.pop.update_positions('global', 'greedy')

    def apply_search_operator(self, perturbator, selector):
        # Split operator
        operator_name, operator_params = perturbator.split('(')

        # Apply an operator
        exec('Operators.' + operator_name + '(self.pop,' + operator_params)

        # Evaluate fitness values
        self.pop.evaluate_fitness(self._problem_function, self.file_name_fitness_values)

        # Update population
        if selector in __selectors__:
            self.pop.update_positions('population', selector)
        else:
            self.pop.update_positions()

        # Update global position
        self.pop.update_positions('global', 'greedy')

    def run(self, hh_step=None, file_label=None):
        """
        Run the metaheuristic for solving the defined problem.
        :parameter int hh_step: Optional
            NOTE: CUSTOM CHANGE BY LARS - Value that passes the hh_step number such that it can save the best positions per agent for the hh step.
        :parameter str label: Optional
            NOTE: CUSTOM CHANGE BY LARS - File label to use for saving design point data.
        :return: None.
        """
        if (not self.perturbators) or (not self.selectors):
            raise Operators.OperatorsError("There are not perturbator or selector!")

        # Apply initialiser / Random Sampling
        self.apply_initialiser(file_label)

        # TODO: Save design points in CSV file.

        # Initialise and update historical variables
        self.reset_historicals()
        self.update_historicals()

        # Report which operators are going to use
        self._verbose('\nSearch operators to employ:')
        for perturbator, selector in zip(self.perturbators, self.selectors):
            self._verbose("{} with {}".format(perturbator, selector))
        self._verbose("{}".format('-' * 50))

        # Start optimisaton procedure
        while not self.finaliser():
            # Update the current iteration
            self.pop.iteration += 1

            # Implement the sequence of operators and selectors
            for perturbator, selector in zip(self.perturbators, self.selectors):

                # Apply the corresponding search operator
                self.apply_search_operator(perturbator, selector)

                # Update historical variables
                self.update_historicals()

            # Verbose (if so) some information
            self._verbose('{}\npop. radius: {}'.format(self.pop.iteration, self.historical['radius'][-1]))
            self._verbose(self.pop.get_state())

            # TODO: Save design points in CSV file.

        # NOTE: CUSTOM CHANGE BY LARS - Save the best position per agent of the finished MH.
        if self.pass_finalised_positions:
            self._save_best_positions_per_agent_to_csv(hh_step, file_label)

    def set_finalisation_conditions(self, conditions):
        # TODO: Check that it works for budgets <=
        if not isinstance(conditions, list):
            conditions = list(conditions)

        self.finalisation_conditions = conditions

    def finaliser(self):
        criteria = self.pop.iteration >= self.num_iterations
        if self.finalisation_conditions is not None:
            for condition in self.finalisation_conditions:
                criteria |= condition()

        return criteria

    def get_solution(self):
        """
        Deliver the last position and fitness value obtained after ``run`` the metaheuristic procedure.
        :returns: ndarray, float
        """
        return self.historical['position'][-1], self.historical['fitness'][-1]

    def reset_historicals(self):
        """
        Reset the ``historical`` variables.
        :return: None.
        """
        self.historical = dict(fitness=list(), position=list(), centroid=list(), radius=list())

    def update_historicals(self):
        """
        Update the ``historical`` variables.
        :return: None.
        """
        # Update historical variables
        self.historical['fitness'].append(np.copy(self.pop.global_best_fitness))
        self.historical['position'].append(np.copy(self.pop.global_best_position))

        # Update population centroid and radius
        current_centroid = np.array(self.pop.positions).mean(0)
        self.historical['centroid'].append(np.copy(current_centroid))
        self.historical['radius'].append(np.max(np.linalg.norm(self.pop.positions - np.tile(
            current_centroid, (self.num_agents, 1)), 2, 1)))

    def _verbose(self, text_to_print):
        """
        Print each step performed during the solution procedure. It only works if ``verbose`` flag is True.
        :param str text_to_print:
            Explanation about what the metaheuristic is doing.
        :return: None.
        """
        if self.verbose:
            print(text_to_print)


    def _save_best_positions_per_agent_to_csv(self, hh_step, file_label):
        """Main function to save best positions per agent to a CSV file."""
        # Ensure directory exists
        best_positions_folder_path = os.path.join(os.getcwd(), "data/raw/design_points/best_positions")
        os.makedirs(best_positions_folder_path, exist_ok=True)

        # Format data into a DataFrame
        best_positions_df = self._format_best_positions_data(hh_step)

        # Define the complete file path
        best_design_points_file_path = os.path.join(best_positions_folder_path, f"{file_label}.csv")

        # Save the DataFrame to CSV
        self._save_dataframe_to_csv(best_positions_df, best_design_points_file_path)


    def _format_best_positions_data(self, hh_step):
        """Format the best positions data into a DataFrame."""
        particular_best_fitness = self.pop.particular_best_fitness
        particular_best_positions = self.pop.particular_best_positions
        agent_ids = range(self.pop.num_agents)

        # Prepare the main columns of the DataFrame
        best_positions = {
            "hh_step": hh_step,
            "agent_id": agent_ids,
            "agent_fitness_value": particular_best_fitness
        }

        # Convert particular_best_positions to a DataFrame with config columns
        n_configs = particular_best_positions.shape[1]  # Number of configuration columns
        config_columns = [f"config_{i+1}" for i in range(n_configs)]
        config_df = pd.DataFrame(particular_best_positions, columns=config_columns)

        # Combine main columns and config columns into one DataFrame
        best_positions_df = pd.DataFrame(best_positions).join(config_df)
        return best_positions_df


    def _save_design_points_mh_iteration_to_csv(self, hh_step, file_label):
        """Main function to save best positions per agent to a CSV file."""
        # Ensure directory exists
        design_points_folder_path = os.path.join(os.getcwd(), "data/raw/design_points/design_points")
        os.makedirs(design_points_folder_path, exist_ok=True)

        # Format data into a DataFrame
        design_points_df = self._format_design_points_data(hh_step)

        # Define the complete file path
        best_design_points_file_path = os.path.join(design_points_folder_path, f"{file_label}.csv")

        # Save the DataFrame to CSV
        self._save_dataframe_to_csv(design_points_df, best_design_points_file_path)


    def _format_design_points_data(self, hh_step):
        """Format the design points data into a DataFrame."""
        design_points_fitness = None
        design_points_positions = None
        agent_ids = range(self.pop.num_agents)

        # Prepare the main columns of the DataFrame
        design_points = {
            "hh_step": hh_step,
            "iteration_id": None,
            "agent_id": agent_ids,
            "agent_fitness_value": None
        }

        # Convert particular_best_positions to a DataFrame with config columns
        n_configs = None
        config_columns = [f"config_{i+1}" for i in range(n_configs)]
        config_df = pd.DataFrame(None, columns=config_columns)

        # Combine main columns and config columns into one DataFrame
        best_positions_df = pd.DataFrame(design_points).join(config_df)
        return best_positions_df


    def _save_dataframe_to_csv(self, df, file_path):
        """Save a DataFrame to a CSV file, creating or appending as needed."""
        if not os.path.isfile(file_path):
            # Create a new file with headers if it doesn't exist
            df.to_csv(file_path, index=False, mode='w', header=True)
        else:
            # Append without headers if the file already exists
            df.to_csv(file_path, index=False, mode='a', header=False)
