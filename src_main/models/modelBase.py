import numpy as np

# from external.CUSTOMHys.customhys import benchmark_func as bf
from customhys import benchmark_func as bf
import customhys


# Import BasicProblem object for generating a custom optimisation problem.
BP = bf.BasicProblem

'''
    Subclass of BasicProblem object to generate a custom optimisation problem.

    variable_num:              Number of dimensions of the problem
    max_search_range:          Maximum boundaries of parameter values
    min_search_range:          Minimum boundaries of parameter values
    optimal_solution:          Optimal configuration of parameter values
    global_optimum_solution:   Optimal fitness value of optimal configuration
    func_name:                 Name of the function?
'''


class instanceBase(BP):
    def __init__(self,
                variable_num,
                max_search_range,
                min_search_range,
                optimal_solution,
                global_optimum_solution,
                func_name,
                sim_run,
                boundaries,
                fitness_value_dir):
        super().__init__(variable_num)
        self.max_search_range = max_search_range
        self.min_search_range = min_search_range
        self.optimal_solution = optimal_solution
        self.global_optimum_solution = global_optimum_solution
        self.func_name = func_name
        self.sim_run = sim_run
        self.features = {'Continuous': False,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

        # Determine min and max values for objectives
        self.boundaries = boundaries
        self.file_name_fitness_values="fitness_values.json"
        self.fitness_value_dir = fitness_value_dir
        self.step_iteration_data = {"problem": None, 'step': -1, 'iteration': -1}


    '''
        Set file name for local fitness values file for running multiple HH/MH at once
    '''
    def set_space_name(self, space_name):
        self.space_name = space_name
        self.step_iteration_data["problem"] = space_name

    def set_file_name_fitness_values(self, file_name_fitness_values):
        self.file_name_fitness_values = file_name_fitness_values

    def get_file_name_fitness_values(self):
        return self.file_name_fitness_values

    '''
        Set which step and iteration is currently evaluated to perform better analysis of cached simulations
    '''
    def set_step_iteration_data(self, step, iteration):
        self.step_iteration_data["step"] = step
        self.step_iteration_data["iteration"] = iteration

    def get_step_iteration_data(self):
        return self.step_iteration_data

    def get_formatted_problem(self, is_constrained=True, fts=None):
        return dict(function=lambda x: self.get_function_value(x),
                    boundaries=(self.min_search_range, self.max_search_range),
                    is_constrained=is_constrained,
                    features=self.get_features(fts=fts),
                    func_name=self.func_name,
                    dimensions=self.variable_num,
                    set_space_name=lambda x: self.set_space_name(x),
                    set_file_name_fitness_values=lambda x: self.set_file_name_fitness_values(x),
                    get_file_name_fitness_values=lambda: self.get_file_name_fitness_values(),
                    fitness_value_dir = self.fitness_value_dir,
                    set_step_iteration_data=lambda x,y: self.set_step_iteration_data(x, y),
                    get_step_iteration_data=lambda: self.get_step_iteration_data()
        )

    '''
        Run the simulation model with the given variables.
    '''
    def get_func_val(self, variables, *args):
        return self.sim_run(self.fitfunc, variables, self.file_name_fitness_values, self.step_iteration_data)


    '''
        Evaluate the fitness value of the simulation run.

        Args:
            simulation_metrics: The simulation metrics obtained from the simulation run.
    '''
    def fitfunc(self, fitness_config, simulations_metrics):
        raise NotImplementedError("You need to implement a fitness function.")

