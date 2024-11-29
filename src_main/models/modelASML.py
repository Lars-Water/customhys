import numpy as np

# from external.CUSTOMHys.customhys import benchmark_func as bf
from customhys import benchmark_func as bf
import customhys

# Import BasicProblem object for generating a custom optimisation problem.
from .modelBase import instanceBase

'''
    Subclass of BasicProblem object to generate a custom optimisation problem.

    variable_num:              Number of dimensions of the problem
    max_search_range:          Maximum boundaries of parameter values
    min_search_range:          Minimum boundaries of parameter values
    optimal_solution:          Optimal configuration of parameter values
    global_optimum_solution:   Optimal fitness value of optimal configuration
    func_name:                 Name of the function?
'''


class instanceASML(instanceBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_iteration_data = {"problem": None, 'step': -5, 'iteration': -5}

    '''
        Evaluate the fitness value of the simulation run.

        Args:
            simulation_metrics: The simulation metrics obtained from the simulation run.
    '''
    def fitfunc(self, fitness_config, simulations_metrics):
        # TODO ASML
        fitness_function = fitness_config["fitness_function"]

        fitness_values = {}
        for simulation_metrics in simulations_metrics:
            (agent_id, metrics), = simulation_metrics.items()
            # Fitness value evaluates the objectives for latency and network cost.
            if fitness_function == "wfpm":
                wfpm = metrics["wfpm"]
                cost = metrics["cost"]

                normalized_wfpm = (wfpm - self.boundaries['wfpm']['min'])/(self.boundaries['wfpm']['max'] - self.boundaries['wfpm']['min'])
                normalized_cost = (cost - self.boundaries['cost']['min'])/(self.boundaries['cost']['max'] - self.boundaries['cost']['min'])

                weight_wfpm = fitness_config["weight_wfpm"]
                weight_cost = fitness_config["weight_cost"]
                fitness_value = (weight_wfpm * normalized_wfpm) + (weight_cost * normalized_cost)
                fitness_values[agent_id] = fitness_value

            # TODO: Add other fitness functions here.
            else:
                return 0
        # self.logger.info("fitfunc fitness_values:"+str(fitness_values))
        return fitness_values