import numpy as np

# from external.CUSTOMHys.customhys import benchmark_func as bf
from src_main.external.customhys_local import benchmark_func as bf
import src_main.external.customhys_local as customhys


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


class instanceINET(instanceBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

