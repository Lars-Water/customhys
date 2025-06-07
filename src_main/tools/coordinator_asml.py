from src_main.tools.logger import logger, setLevelLogger
from src_main.data.collect_data_ASML import DataCollectorASML
from src_main.tools.hyperheuristicBase import HyperHeuristicBase
import src_main.tools.component_config as component_config
import src_main.tools.file_operations as fo


import os
import glob
import shutil
import re
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
import glob
import itertools
import xml.etree.ElementTree as ET
import rich.progress as prog

from experiments import create_sim_asml, create_sim_asml_parallel
from src.manager import Manager
from src.utils.config_reader import Config
from src.utils.config_creator import WorkflowConfig

import uuid

from .coordinatorBase import HeuristicSimulationCoordinatorBase
from src_main.tools.local_parallelization.rpc import requires_main_process, callable_from_main


class HeuristicSimulationCoordinatorASML(HeuristicSimulationCoordinatorBase):

    def __init__(self, base_path, coordinator_config_file_path, nr_of_agents, run_name=None, nr_of_design_queues=0, experiment_config=None, normalize=True):
        super().__init__(base_path, coordinator_config_file_path, nr_of_agents, run_name=run_name, nr_of_design_queues=nr_of_design_queues, experiment_config=experiment_config, normalize=normalize)
        self.setNumberOfCoresWithFrequencies()

        self.min_wfpm_runtime = 157455.0
        self.max_wfpm_runtime = 179489.0
        self.min_cost = -23.0
        self.max_cost = 46.0

        # self.setNumberOfCoresWithActive()

    def problemInstanceFunc(self):
        return component_config.create_problem_instanceASML

    @requires_main_process
    def createDataCollector(self):
        weight_wfpm = self.conf.tryGet("fitness_config", "weight_wfpm")
        weight_cost= self.conf.tryGet("fitness_config", "weight_wfpm")
        self.data_collector = DataCollectorASML(weight_wfpm, weight_cost, self.dir_design_points_metrics_output, run_name=self._run_name )
    
    @requires_main_process
    def createHyperHeuristicBase(self):
        self.template_xml_file_path = os.path.join(self.simulation_model_template_path, "platform.xml")
        self.hh_base = HyperHeuristicBase(
            self,
            self._base_path,
            self.coordinator_config_file_path,
            self._experiment_config,
            self.template_xml_file_path,
            self.log_path,
            "ASML-Faezeh",
            "experiments_asml",
            self._run_name
        )

    def setNumberOfCoresWithFrequencies(self):
        tree = ET.parse(self.template_xml_file_path)
        cores = tree.findall('.//core[@frequency]')
        self.numberOfCoresWithFrequencies = len(cores)

    # def setNumberOfCoresActive(self):
    #     self.numberOfCoresActive = 0
    #     simulation_model_params = self.conf.tryGet("simulation_model", "simulation_model_params")
    #     for sim_param in sim_param:
    #         if sim_param["param_name"].include("processor_cores_active"):
    #             self.numberOfCoresActive += len(sim_param["configuration_pattern"])        

    def generate_design_point(self, sim_id, agent_configuration):
        """
        Generates a simulation instance for a given design point.

        Args:
            sim_id (int): The ID of the design point.
            agent_configuration (dict): The configuration of the agent.

        Returns:
            None
        """
        design_point_path = self.generated_simulation_model_path + f"_{sim_id}"

        # TODO: Change the hardcoded ini filename to a variable in the configuration file.
        # Duplicate the preferred dummy_sim directory to the custom directory.
        self.duplicate_directory(self.simulation_model_template_path, design_point_path, files_to_ignore=["platform.xml", "*.pstat", "*.cc", "*.msg", "*.h", "*.csv", "*.log", "out/*", ".*", ".settings/", "out", "*.o"]) # , "results/*"

        # TODO: Change the hardcoded ini filename to a variable in the configuration file.
        # Write an updated version of the ignored param value file from the directory that was just duplicated.
        design_point_ini_file_path = os.path.join(design_point_path, "platform.xml")

        # TODO: Change the hardcoded number of backbone switches to a variable.
        # Write a new ini file with the parameter configurations.
        component_config.generate_asml_config(
            self.template_xml_file_path,
            design_point_ini_file_path,
            agent_configuration
        )


    '''
    Set the values of the parameters in the simulation model.

    Args:
        config_values: A list of values for the parameters in the simulation model.
    '''
    def set_param_values(self, config_values):
        self.logger.info("Setting up the simulation model parameters...")
        params = self.conf.tryGet("simulation_model", "simulation_model_params")

        # Transform input values into configuration parameters.
        configurations = []

        raise Error("Please configure more than 1 agent.")
        
        return configurations

    '''
    Set the values of the parameters of multiple agents in the simulation model.

    Args:
        config_values: A list of values for the parameters in the simulation model.
    '''
    def set_agents_param_values(self, config_values_agents):
        params = self.conf.tryGet("simulation_model", "simulation_model_params")
        configurations = []

        self.logger.debug(f"Config values agents: {config_values_agents}")

        for config_values_agent in config_values_agents:
            self.logger.debug(f"Configuring agent : {config_values_agent}")
            agent_configuration = []

            for param in params:
                self.logger.debug(f"Configuring param : {param}")
                config_pattern = param["configuration_pattern"]
                param_values = param["values"]

                # Determine the param value index by segment indexing the config values from CUSTOMHys.
                self.logger.debug(f"Determine the param value index")
                num_segments = len(param_values)
                value_indices = np.floor(config_values_agent * num_segments).astype(int)
                value_indices[config_values_agent == 1.0] = num_segments - 1

                # TODO: Optimize this operations as it now parses and has duplicate code.
                # Create a configuration for each parameter.
                self.logger.debug(f"Define the correct gate and switch indices for the parameter: {value_indices}")
                for core_idx, value_idx in enumerate(value_indices):
                    selected_param_value = param_values[value_idx]
   
                    # Replace placeholders in the configuration pattern are replaced with the corresponding indices.
                    current_param_pattern = config_pattern[:]
                    current_param_pattern.append(core_idx)
                    # print("### ", current_param_pattern)

                    # Create a configuration for the current parameter.
                    agent_configuration.append({
                        "param_name": param["param_name"],
                        "config_pattern": current_param_pattern,
                        "value": selected_param_value
                    })

            self.logger.debug(f"Append agent configuration: {agent_configuration}")
            configurations.append(agent_configuration)

        return configurations


    '''
        Collect the simulation stats from the simulation run.

        Args:
            uid: The unique identifier of the simulation run.
    '''
    def obtain_simulation_stats(self, uids, file_name_fitness_values="fitness_values.json"):
        # TODO: Should rename fitness_values variable accross the board to something else, because this is not the correct term.
        # Load the transformed outputted data from the simulation run.
        fitness_values = []

        for sim_uid, agent_id in uids.items():

            # TODO: Change scavetool output filename to something more descriptive.
            folder_path = os.path.join(self.data_path, "results", sim_uid)
            csv_file_path = os.path.join(folder_path, "x.csv")

            # Todo: Check for raceconditions while results files are written by the workers, until then just wait before reading the file
            i_trys = 0
            while True and i_trys <= 1000:
                try:
                    df = pd.read_csv(csv_file_path)
                    pass 
                    # if this point is reached everything worked fine, so exit loop
                    break
                except:
                    i_trys += 1
                    self.logger.error("Error reading "+str(csv_file_path)+" (Try #"+str(i_trys)+"), TRYING AGAIN IN 200MS.")
                    # os.listdir(csv_file_path_dbg)
                    time.sleep(0.2)

            # Try one last time to create an intentional FileNotFoundError for the user
            if (i_trys >= 1000): 
                df = pd.read_csv(csv_file_path)
            
            # print("########################")
            # self.logger.info(f"DF: {df}")
            # Determine end-to-end delay statistics.
            simtime_df = df[df["name"].fillna("").str.endswith("#waverage")]
            wfpm = pd.to_numeric(simtime_df.loc[simtime_df['value'].idxmax()]['value'], downcast='float')

            cost_df = df[df["name"].fillna("").str.endswith("#cost")]
            cost = pd.to_numeric(cost_df['value'], downcast='float').sum()

            # Store design point metrics output.
            if self.store_design_points_metrics_values:
                self.try_load_problems_file_name_fitness_values(file_name_fitness_values)
                self.logger.debug(f"Storing metrics output values for siminstance {sim_uid}.")
                self.data_collector.cache_design_point_metrics_collective(wfpm, cost, sim_uid, file_name_fitness_values)

            # Remove the csv file.
            if self.remove_sim_instance_output:
                self.remove_sim_instance_output_files(sim_uid, csv_file_path, folder_path)

            fitness_values.append({
                agent_id: {
                    "wfpm": wfpm,
                    "cost": cost
                }
            })
        
        self.data_collector.store_cached_design_point_metrics_collective(file_name_fitness_values)

        # return the average of the column 'mean' in the dataframe.
        return fitness_values


    def get_boundaries(self):
        return self.min_wfpm_runtime, self.max_wfpm_runtime, self.min_cost, self.max_cost

    @requires_main_process
    def manual_normalization(self):
        """
        Manually determines the min and max values for the objective parameters.

        This method generates simulation instances with different parameter values
        based on the specified boundaries. It creates a unique simulation instance
        for each combination of parameter and value, and stores the generated simulation
        IDs in a list.

        Returns:
            None
        """
        self.logger.info("Manually determine the min and max values for the objective parameters.")

        # TODO: Remove hardcoded filter for ignoring cable colours? Depends on if I change the actualy configuration and parameter setting functionality.
        # Determine the minimal and maximal parameter values for the simulation model parameters.
        with prog.Progress(
                prog.SpinnerColumn(),
                prog.TextColumn("[progress.description]{task.description}", justify="right"),
                prog.TextColumn("[progress.percentage]{task.completed}/{task.total}", justify="right"),
                prog.BarColumn(),
                prog.TimeElapsedColumn(),
        ) as progress:
            simulation_model_params = self.conf.tryGet("simulation_model", "simulation_model_params")
            boundaries = {}
            for ids, simulation_model_param in enumerate(simulation_model_params):
                ignore_normalization = simulation_model_param['ignore_normalization']
                if not ignore_normalization:
                    boundaries[simulation_model_param['param_name']+str(ids)] = {
                        "values": {
                            "max": simulation_model_param['values'][-1],
                            "min": simulation_model_param['values'][0]
                        },
                        "configuration_pattern": simulation_model_param['configuration_pattern']
                    }
            print(boundaries)
            # Generate simulation instances the min and max parameter value configurations.
            sim_ids = []
            sim_id_boundaries = {}
            sim_id_bars = {}
            # print("##### boundaries:")
            # print(boundaries)
            tree = ET.parse(self.template_xml_file_path)
            for parameter, boundaries_local in boundaries.items():
                sim_id_boundaries[parameter] = {}
                sim_id_bars[parameter] = {}
                for boundary in boundaries_local["values"]:
                    value = boundaries_local["values"][boundary]
                    self.logger.debug(f"Generating normalization sim instance for parameter: {parameter} - boundary: {boundary} - value: {value}")
                    sim_id = uuid.uuid4()
                    bar = progress.add_task(
                        f"[bold]{parameter}[/bold] {boundary}", 
                        total=1,
                        start=True,
                        completed=-0
                    )
                    if parameter.startswith("processor_freq"):
                        configuration = [
                            {
                                "param_name": parameter,
                                "config_pattern": [boundaries_local['configuration_pattern'][0], x, "frequency"],
                                "value": value
                            } for x in range(self.numberOfCoresWithFrequencies)
                        ]
                        if boundary == "max":
                            configuration += [{
                                "param_name":  "processor_freq",
                                "config_pattern": [".//core[@active]", x, "active"],
                                "value": ["true", 1]
                            } for x in range(len(tree.findall('.//core[@active]')))]
                    elif parameter.startswith("num_processor_cores_active"):
                        configuration = [
                            {
                                "param_name": parameter,
                                "config_pattern": [boundaries_local['configuration_pattern'][0], 0],
                                "value": value
                            }
                        ]
                    self.generate_design_point(sim_id, configuration)
                    sim_ids.append(sim_id)
                    sim_id_boundaries[parameter][boundary] = sim_id
                    sim_id_bars[parameter][boundary] = bar

            self.logger.info("Evaluating the generated simulation instances.")
            uids = self.run_multiple_simulation_configuration(sim_ids, step_iteration_data = ["manual_normalization"])

            for parameter, boundaries_local in boundaries.items():
                for boundary in boundaries_local["values"]:
                    value = boundaries_local["values"][boundary]
                    progress.advance(sim_id_bars[parameter][boundary])
                    uid = list(uids.keys())[list(uids.values()).index(sim_ids.index(sim_id_boundaries[parameter][boundary]))]
                    val_boun = self.determine_boundary_value(uid, parameter, boundary)
                    self.logger.debug(f"Determine the boundary value for the objective: {parameter} - boundary: {boundary} - value: {value} --> {val_boun}")

            sim_ids.clear()

            if self.remove_design_point_configuration_dummy_path:
                self.logger.debug("Removing simulation run templates in dummy path.")
                fo.remove_design_point_configurations_dummy_path(self.generated_path, pattern=self.remove_design_point_configuration_dummy_path_pattern+"*")
                
            if self.remove_sim_instance_experiments_folder:
                self.logger.debug("Removing simulation instances from experiments folder.")
                fo.remove_sim_instance_folders(self.data_path, uids,
                                                self.remove_sim_instance_experiments_folder["logs"], 
                                                self.remove_sim_instance_experiments_folder["results"], 
                                                self.remove_sim_instance_experiments_folder["runtime"]
                )
            
                
            self._check_normalization()
            self.logger.debug("Boundaries: \n" +
                "self.min_wfpm_runtime: " + str(self.min_wfpm_runtime) + "\n" +
                "self.max_wfpm_runtime: " + str(self.max_wfpm_runtime) +"\n" +
                "self.min_cost: " + str(self.min_cost) + "\n" +
                "self.max_cost: " + str(self.max_cost) 
            )


    @requires_main_process
    def determine_boundary_value(self, sim_uid, parameter, boundary):
    # TODO ASML
        csv_file_path = os.path.join(self.data_path, "results", str(sim_uid), "x.csv")
        df = pd.read_csv(csv_file_path)

        # Determine boundary values for the datarate parameter.
        if parameter.startswith("processor_freq") or parameter.startswith("num_processor_cores_active"):
            simtime_df = df[df["name"].fillna("").str.endswith("#waverage")]
            simtime_max = float(simtime_df.loc[simtime_df['value'].idxmax()]['value'])

            cost_df = df[df["name"].fillna("").str.endswith("#cost")]
            # print(cost_df)
            cost = float(pd.to_numeric(cost_df['value']).sum())

            # print(parameter, boundary, simtime_max, cost)
            # if boundary == "min":
            if hasattr(self, "min_wfpm_runtime"):
                self.min_wfpm_runtime = min(self.min_wfpm_runtime, simtime_max) # wfpm_runtime
            else:
                self.min_wfpm_runtime = simtime_max
            if hasattr(self, "min_cost"):
                self.min_cost = min(self.min_cost, cost) # wfpm_runtime
            else:
                self.min_cost = cost
            # elif boundary == "max":
            if hasattr(self, "max_wfpm_runtime"):
                self.max_wfpm_runtime = max(self.max_wfpm_runtime, simtime_max) #wfpm_runtime
            else:
                self.max_wfpm_runtime = simtime_max
            if hasattr(self, "max_cost"):
                self.max_cost = max(self.max_cost, cost) #wfpm_runtime
            else:
                self.max_cost = cost
    

    def _check_normalization(self):
    # TODO ASML
        if hasattr(self, "min_wfpm_runtime"):
            self.logger.info(f"Minimum WFPM is: {self.min_wfpm_runtime}")
        else:
            raise ValueError("Normalisation went wrong, min_wfpm_runtime values are missing.")

        if hasattr(self, "max_wfpm_runtime"):
            self.logger.info(f"Maximum WFPM is: {self.max_wfpm_runtime}")
        else:
            raise ValueError("Normalisation went wrong, max_wfpm_runtime values are missing.")
            
        if hasattr(self, "min_cost"):
            self.logger.info(f"Minimum Cost is: {self.min_cost}")
        else:
            raise ValueError("Normalisation went wrong, min_cost values are missing.")
                        
        if hasattr(self, "max_cost"):
            self.logger.info(f"Maximum Cost is: {self.max_cost}")
        else:
            raise ValueError("Normalisation went wrong, max_cost values are missing.")

    def create_dummy(self, *args, **kwargs):
        return create_sim_asml(*args, **kwargs)

    def create_dummy_parallel(self, *args, **kwargs):
        return create_sim_asml_parallel(*args, **kwargs)