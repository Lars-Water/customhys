import numpy as np

# from external.CUSTOMHys.customhys import benchmark_func as bf
from customhys import benchmark_func as bf

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
                 sim_run):
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
                         'Unimodal': True,
                         'Convex': False}

    '''
        Run the simulation model with the given variables.
    '''
    def get_func_val(self, variables, *args):
        return self.sim_run(variables)


'''
    Generate a basic problem instance from a given simulation model
    configuration and simulation run functionality.
'''


def generate_instance(variable_num, instance_config, sim_run):
    min_range = np.array([instance_config['boundaries'][key][0]
                          for key in instance_config['boundaries']])
    max_range = np.array([instance_config['boundaries'][key][1]
                          for key in instance_config['boundaries']])

    return instance(variable_num,
                    min_range,
                    max_range,
                    instance_config['optimal_solution'],
                    instance_config['optimal_fitness'],
                    'CQN',
                    sim_run)
