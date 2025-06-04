import os
import time
import threading
from pathlib import Path

from src_main.external.customhys.customhys import hyperheuristic as hh

from src_main.data import collect_data_INET
from src_main.tools import component_config
from src_main.tools import coordinator_asml
from src_main.tools.config_reader import Config

save_runs = []
experiment_config = None
max_cost = None
def run_experiment(experiment_config, coordinator_params):
    '''
        Experimental run of tuning the parameters of any provided search operators.
    '''
    search_operators_spaces = experiment_config.tryGet('search_operators', 'spaces')
    pass_finalised_positions = experiment_config.tryGet('pass_finalised_positions')
    hh_parameters = experiment_config.tryGet('hh_parameters')
    num_replicas = hh_parameters["num_replicas"] if hh_parameters["num_replicas"] > 0 else 1 

    
    # Create the HeuristicSimulationCoordinator.
    base_path, coordinator_config_file_path, nr_of_agents, run_name = coordinator_params
    heur_sim_coordinator = coordinator_asml.HeuristicSimulationCoordinatorASML(
        base_path, 
        coordinator_config_file_path, 
        nr_of_agents, 
        run_name=run_name,
        nr_of_design_queues = len(search_operators_spaces.keys())*num_replicas,
        experiment_config=experiment_config,
        normalize=True
    ) # noqa 501
    heur_sim_coordinator.set_run_name("ASML_"+str(heur_sim_coordinator.timestamp))

    heur_sim_coordinator.run()
