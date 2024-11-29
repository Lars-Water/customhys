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


class instance(BP):
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


    '''
        Set file name for local fitness values file for running multiple HH/MH at once
    '''
    def set_file_name_fitness_values(self, file_name_fitness_values):
        self.file_name_fitness_values = file_name_fitness_values

    def get_file_name_fitness_values(self):
        return self.file_name_fitness_values


    def get_formatted_problem(self, is_constrained=True, fts=None):
        return dict(function=lambda x: self.get_function_value(x),
                    boundaries=(self.min_search_range, self.max_search_range),
                    is_constrained=is_constrained,
                    features=self.get_features(fts=fts),
                    func_name=self.func_name,
                    dimensions=self.variable_num,
                    set_file_name_fitness_values=lambda x: self.set_file_name_fitness_values(x),
                    get_file_name_fitness_values=lambda: self.get_file_name_fitness_values(),
                    fitness_value_dir = self.fitness_value_dir
        )

    '''
        Run the simulation model with the given variables.
    '''
    def get_func_val(self, variables, *args):
        return self.sim_run(self.fitfunc, variables, self.file_name_fitness_values)


    '''
        Evaluate the fitness value of the simulation run.

        Args:
            simulation_metrics: The simulation metrics obtained from the simulation run.
    '''
    def fitfunc(self, fitness_config, simulations_metrics):
        fitness_function = fitness_config["fitness_function"]

        fitness_values = {}
        for simulation_metrics in simulations_metrics:
            (agent_id, metrics), = simulation_metrics.items()
            # Fitness value evaluates the objectives for latency and network cost.
            if fitness_function == "latency_cost":
                latency = metrics["latency"]
                network_cost = metrics["network_cost"]
                normalized_latency = (latency - self.boundaries['datarate']['min'])/(self.boundaries['datarate']['max'] - self.boundaries['datarate']['min'])
                normalized_network_cost = (network_cost - self.boundaries['cost']['min'])/(self.boundaries['cost']['max'] - self.boundaries['cost']['min'])
                weight_latency = fitness_config["weight_latency"]
                weight_cost = fitness_config["weight_cost"]
                fitness_value = (weight_latency * (1 - normalized_latency)) + (weight_cost * normalized_network_cost)
                fitness_values[agent_id] = fitness_value

            # TODO: Add other fitness functions here.
            else:
                return 0
        # self.logger.info("fitfunc fitness_values:"+str(fitness_values))
        return fitness_values

