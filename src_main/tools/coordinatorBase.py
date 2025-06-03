from src_main.tools.logger import logger, setLevelLogger
import src_main.tools.component_config as component_config
import src_main.tools.file_operations as fo
from src_main.tools.config_reader import Config
from src_main.tools.problemSpace import ProblemSpace
from src_main.tools.searchOperatorSpace import SearchOperatorSpace
from src_main.tools.hyperheuristicBase import HyperHeuristicBase

from src_main.data import collect_data_INET

import os
import glob
import shutil
import re
import pandas as pd
import numpy as np
import time
import json
import sys
from pathlib import Path
import glob
import itertools
from threading import Lock
import math


from experiments import create_sim_custom_dummy, create_sim_inet_lans_dummy, create_sim_inet_lans_dummy_parallel
from src.manager import Manager
from src.utils.config_creator import WorkflowConfig
from src_main.external.customhys.customhys import hyperheuristic as hh

import uuid


class HeuristicSimulationCoordinatorBase:
    hh_base = None

    def __del__(self):
        if hasattr(self, "manager"):
            self.manager.shutdown()

    '''
        Initialize the HeuristicSimulationCoordinator.

        Args:
            base_path: The base path of the project.
            coordinator_config_file_path: The path to the configuration file of the coordinator.
            nr_of_agents: The number of agents to run the simulation model with.
    '''
    def __init__(self, base_path, coordinator_config_file_path, nr_of_agents, run_name=None,  nr_of_design_queues=0, experiment_config=None, normalize=True): 
        self._base_path = base_path
        self._nr_of_agents = nr_of_agents
        self._nr_of_sims = nr_of_agents
        self._nr_of_design_queues = nr_of_design_queues
        self._run_name = run_name
        self._normalize = normalize
        if experiment_config is not None:
            self._experiment_config = experiment_config

        # Predefine the parameters for the execution times.
        self.coordinator_and_simulation_execution_time = 0
        self.simulation_execution_time = 0

        self.uids = []
        self.queues = {}

        self.coordinator_config_file_path = coordinator_config_file_path
        self.conf = Config(coordinator_config_file_path, name = "coordinator_config_manager")
        self.log_path = self.conf.tryGet("output_paths", "log_files")
        if self.log_path is None:
            self.log_path = os.path.join(self._base_path, "data/logs/")

        coordinator_log_path = Path(os.path.join(self.log_path, "coordinator"))
        self.conf.createLogger(coordinator_log_path, "coordinator_config_manager")
        self.logger = logger("coordinator", coordinator_log_path, disabled=False)

        # Set logger level.
        logger_lvl = self.conf.tryGet("logger_lvl")
        setLevelLogger(self.logger, logger_lvl)

        # Define params to set up the Manager.
        experiments_path = self.conf.tryGet("simulation_model", "simulation_model_paths", "experiments_path")
        sim_model = self.conf.tryGet("simulation_model", "simulation_model_configuration", "sim_model")
        num_nodes = self.conf.tryGet("simulation_model", "simulation_model_configuration", "num_nodes")
        num_workers = self.conf.tryGet("simulation_model", "simulation_model_configuration", "num_workers")
        if self.conf.tryGet("simulation_model", "simulation_model_configuration", "platform") == "DAS":
            num_nodes = self.conf.tryGet("simulation_model", "simulation_model_configuration", "jobs")
            num_workers = self.conf.tryGet("simulation_model", "simulation_model_configuration", "job_cores")*self.conf.tryGet("simulation_model", "simulation_model_configuration", "job_processes")

        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.timestamp_int = int(time.time()) 
        self.data_path = os.path.join(experiments_path, "data", f"campaign_{sim_model}", f"n{str(num_nodes)}_w{str(num_workers)}_s{str(self._nr_of_sims)}", self.timestamp)

        # Define params for configuration file creation.
        workflow_config_file = os.path.join(self.data_path, "config.json")
        workflow_results_folder = os.path.join(self.data_path, "results")
        workflow_logs_folder = os.path.join(self.data_path, "logs")
        workflow_runtime_folder = os.path.join(self.data_path, "runtime")
        
        self.files_to_keep = []
        if self.conf.tryGet("coordinator_functionalities", "sim_instance_output_files_to_keep"):
            self.files_to_keep = self.conf.tryGet("coordinator_functionalities", "sim_instance_output_files_to_keep")
            
        self.cached_files_evaluation = []
        if self.conf.tryGet("coordinator_functionalities", "sim_instance_cached_files_evaluation"):
            self.cached_files_evaluation = self.conf.tryGet("coordinator_functionalities", "sim_instance_cached_files_evaluation")
            
        self.cache_only_finished_sim_instances=True
        if self.conf.tryGet("coordinator_functionalities", "sim_instance_cached_files_evaluation"):
            self.cache_only_finished_sim_instances = self.conf.tryGet("coordinator_functionalities", "cache_only_finished_sim_instances")

        self.logger.info("Setting up workflow configuration file.")
        self.sims_path = self.conf.tryGet("simulation_model", "simulation_model_paths", "sims_path")
        design_queues = [WorkflowConfig.create_design_point_queue_config(f'q{queue_id}', 0 if queue_id > 0 else 1, "FIFO") for queue_id in range(nr_of_design_queues+1)]
        cluster_config = self.create_relevant_cluster_config()
        self.workflow_config = WorkflowConfig(
            self.sims_path, "config", "run_sim", "results", "logs", "out",
            workflow_results_folder, workflow_logs_folder, workflow_runtime_folder, 
            design_queues, "uuid", cluster_config,  
            remove_sca=False, 
            files_to_keep=self.files_to_keep,
            cached_files_evaluation = self.cached_files_evaluation,
            cache_only_finished_sim_instances = self.cache_only_finished_sim_instances,)
        self.config = self.workflow_config.conf()
        self.workflow_config.write_conf(workflow_config_file)

        self.logger.info("Define the paths for the simulation model results.")
        self.results_path = self.conf.tryGet("results_path")

        self.logger.info("Setting up the Manager.")
        self.manager = Manager(
            workflow_config_file, 
            workflow_logs_folder, 
            stats_file = os.path.join(self.results_path, self._run_name + "_stats.json")
        ) 


        # Define variables for coordinator functionalities.

        self.logger.info("Define the base paths for the simulation model instance generation functionalites.")
        self.simulation_model_template_path = self.conf.tryGet("simulation_model", "simulation_model_paths", "simulation_model_template_path")
        self.ini_file_template_path = os.path.join(self.simulation_model_template_path, "largeNet.ini")
        self.generated_simulation_model_path = self.conf.tryGet("simulation_model", "simulation_model_paths", "simulation_model_generated_path")
        self.inet_path = self.conf.tryGet("simulation_model", "simulation_model_paths", "inet_path")
        self.generated_path = self.conf.tryGet("simulation_model", "simulation_model_paths", "generated_path")

        self.logger.info("Determine flags for coordinator functionalities.")
        self.store_design_points_metrics_values = self.conf.tryGet("coordinator_functionalities", "store_design_points_metrics_values")
        self.remove_sim_instance_output = self.conf.tryGet("coordinator_functionalities", "remove_sim_instance_output")
        self.remove_sim_instance_experiments_folder = self.conf.tryGet("coordinator_functionalities", "remove_sim_instance_experiments_folder")
        self.remove_design_point_configuration_dummy_path = self.conf.tryGet("coordinator_functionalities", "remove_design_point_configuration_dummy_path")
        self.remove_design_point_configuration_dummy_path_pattern = self.conf.tryGet("coordinator_functionalities", "remove_design_point_configuration_dummy_path_pattern") 
        self.remove_design_point_configuration_sims_path = self.conf.tryGet("coordinator_functionalities", "remove_design_point_configuration_sims_path")

        # TODO: Move more functionalities to data collector.
        # TODO: Define config file with a distinct segment for data collector?
        self.logger.info("Define the base paths and variables for simulation instance output functionalities.")
        self.agents_fitness_dir_path = self.conf.tryGet("output_paths", "agents_fitness_values_path")
        self.dir_design_points_metrics_output = self.conf.tryGet("output_paths", "design_points_metrics_output")
        if self.dir_design_points_metrics_output is None or not self.dir_design_points_metrics_output:
            self.dir_design_points_metrics_output = os.path.join(self.data_path, "design_points_metrics")

        self.createDataCollector()
        self.createHyperHeuristicBase()
        self.manager.set_data_collector(self.data_collector)

        self.problems_file_name_fitness_values = {}



        # clear out old agent finess files
        shutil.rmtree(os.path.join(self._base_path, self.agents_fitness_dir_path), ignore_errors=True)

    def createDataCollector(self):
        raise NotImplementedError("You need to implement a data collector creator (createDataCollector()).")

    def createHyperHeuristicBase(self):
        raise NotImplementedError("You need to implement a HyperHeuristic Base (createHyperHeuristicBase()).")

    def problemInstanceFunc(self):
        raise NotImplementedError("You need to set a problem instance creator (problemInstanceFunc()) which returns a problem instance function.")

    def run(self):
        if self._normalize:
            self.logger.info("Start manual normalization.")
            self.manual_normalization()
        else:
            self.logger.warn("Normalization is skipped!")
        self.coordinator_and_simulation_execution_time = 0
        self.simulation_execution_time = 0
        self.problems_file_name_fitness_values = self.hh_base.get_all_problems_file_name_fitness_values()
        return self.hh_base.run_multi_threaded()

    def set_run_name(self, run_name):
        """
        Sets the name of the run.

        Parameters:
        - run_name (str): The name of the run.
        """
        if not isinstance(run_name, str):
            raise ValueError("Run name must be a string.")
        self._run_name = run_name
        self.manager.set_stats_file(os.path.join(self.results_path, self._run_name + "_stats.json")) 
        self.logger.info(f"Run name set to {self._run_name}")


    def create_relevant_cluster_config(self):
        """
        Creates a relevant cluster configuration based on the platform specified in the coordinator configuration.

        Returns:
            A cluster configuration object based on the platform specified in the coordinator configuration.

        Raises:
            ValueError: If an unknown platform value is provided in the coordinator configuration.
        """
        platform = self.conf.tryGet("simulation_model", "simulation_model_configuration", "platform")
        if platform == "DAS":
            self.logger.info("Setting up DAS cluster configuration.")
            return WorkflowConfig.create_slurm_cluster_config(
                self.conf.tryGet("simulation_model", "simulation_model_configuration", "jobs"),
                self.conf.tryGet("simulation_model", "simulation_model_configuration", "job_cores"),
                self.conf.tryGet("simulation_model", "simulation_model_configuration", "job_processes"),
                self.conf.tryGet("simulation_model", "simulation_model_configuration", "job_memory"),
                self.conf.tryGet("simulation_model", "simulation_model_configuration", "walltime")
            )
        elif platform == "local":
            self.logger.info("Setting up local cluster configuration.")
            return WorkflowConfig.create_local_cluster_config(
                self.conf.tryGet("simulation_model", "simulation_model_configuration", "num_workers"),
                self.conf.tryGet("simulation_model", "simulation_model_configuration", "threads_per_worker")
            )
        else:
            raise ValueError(f"Unknown platform value {platform} provided in coordinator config.")


    def generate_design_point(self, sim_id, agent_configuration):
        raise NotImplementedError("You need to implement a design point generator (generate_design_point()).")


    def locally_store_agents_fitness_values(self, fitness_values, file_name_fitness_values="fitness_values.json"):
        """
        Locally caches the fitness values of agents in memory.
        This method previously stored values in a JSON file but has been changed
        to use an in-memory dictionary to avoid frequent file I/O.

        Args:
            fitness_values (dict): A dictionary containing the fitness values of agents.
            file_name_fitness_values (str): Identifier for this set of fitness values.
                                         Its file extension (e.g., .json) will be removed before use as the key.
                                         Defaults to "fitness_values.json" for compatibility with the previous signature.

        Returns:
            None
        """
        # Ensure the in-memory store is initialized.
        # Ideally, this should be in the __init__ method of the class,
        # but this provides a fallback.
        if not hasattr(self.problems_file_name_fitness_values, file_name_fitness_values):
            raise ValueError(f"[locally_store_agents_fitness_values] File name fitness values {file_name_fitness_values} not found in {self.problems_file_name_fitness_values}.")
        # Use the filename without extension as the key
        # memory_key, _ = os.path.splitext(file_name_fitness_values)

        # if hasattr(self, 'logger') and self.logger:
        #     self.logger.info(f"Caching fitness values in memory for key: '{memory_key}' (derived from '{file_name_fitness_values}')")
        
        # self._in_memory_fitness_values[memory_key] = fitness_values
        self.problems_file_name_fitness_values[file_name_fitness_values].store_agents_fitness_values(fitness_values)


    '''
        Run the simulation model with the given configuration values.

        Args:
            fitfunc: The fitness function to evaluate the simulation run.
            config_values: A list of values for the parameters in the simulation model.

        Returns:
            The fitness value of the simulation run.
    '''
    def simulation_run(self, fitfunc, config_values, file_name_fitness_values="fitness_values.json", step_iteration_data={'step': -1, 'iteration': -1}):

        # Start timer for simulation run.
        start_time = time.time()
        # TODO: Merge the following two if statements into one. -> CUSTOMHys should be able to handle both situations?
        if self._nr_of_agents > 1:
            self.logger.info(f"Determining the simulation model parameters for the configuration values.")
            configurations = self.set_agents_param_values(config_values)
            sim_ids = []
            self.logger.info("Generating sim instances for the agents.")
            for agent_configuration in configurations:
                sim_id = uuid.uuid4()
                sim_ids.append(sim_id)
                self.logger.debug(f"Generating simulation instance for design point: {sim_id}.")
                self.generate_design_point(sim_id, agent_configuration)

            self.logger.info("Run the generated simulation instances. ("+str(len(sim_ids))+")\n"+str(file_name_fitness_values)+"\n"+str(step_iteration_data))
            uids = self.run_multiple_simulation_configuration(sim_ids, file_name_fitness_values, step_iteration_data = step_iteration_data)

            self.logger.debug("Collect the simulation stats from the simulation instances runs.\n"+str(uids.keys()))
            simulation_metrics = self.obtain_simulation_stats(uids)

            self.logger.info("Locally storing the agents fitness values.")
            fitness_config = self.conf.tryGet("fitness_config")
            fitness_values = fitfunc(fitness_config, simulation_metrics)

            if self.store_design_points_metrics_values:
                self.data_collector.append_fitness_and_hh_data_to_design_point_metrics(
                    uids, 
                    fitness_values,
                    step_iteration_data["problem"] if "problem" in step_iteration_data else None,
                    step_iteration_data["step"],
                    step_iteration_data["iteration"]
                )
            
            self.locally_store_agents_fitness_values(fitness_values, file_name_fitness_values)

            # End timer for simulation run.
            end_time = time.time()

            # Add simulation run time to total coordinator time.
            self.coordinator_and_simulation_execution_time += end_time - start_time

            if self.remove_design_point_configuration_dummy_path: 
                self.logger.debug("Removing simulation run templates in dummy path.")
                for sim_id in sim_ids:
                    fo.remove_design_point_configurations_dummy_path(self.generated_path, pattern=self.remove_design_point_configuration_dummy_path_pattern+str(sim_id)+"*")
            if self.remove_design_point_configuration_sims_path:
                self.logger.debug("Removing simulation run templates in sims path.")
                # print("sims_path "+str(self.sims_path))
                if self.sims_path and self.sims_path is not None:
                    for sim_id in sim_ids:
                        fo.remove_design_point_configurations_dummy_path(self.sims_path, pattern=str(sim_id))
                # fo.remove_design_point_configurations_sims_path(self.sims_path)
            if self.remove_sim_instance_experiments_folder:
                self.logger.debug("Removing simulation instances from experiments folder.")
                fo.remove_sim_instance_folders(self.data_path, uids,
                                               self.remove_sim_instance_experiments_folder["logs"], 
                                               self.remove_sim_instance_experiments_folder["results"], 
                                               self.remove_sim_instance_experiments_folder["runtime"]
                                            )

            sim_ids.clear()
            return 0

        else:
            # Set the values of the parameters in the simulation model.
            configurations = self.set_param_values(config_values)

            sim_id = uuid.uuid4()
            self.generate_design_point(sim_id, configurations)

            # Run the simulation model.
            uid = self.run_single_simulation_configuration(step_iteration_data = step_iteration_data)

            # Collect the simulation stats from the simulation run.
            simulation_metrics = self.obtain_simulation_stats(uid)

            fitness_config = self.conf.tryGet("fitness_config")
            fitness_value = fitfunc(fitness_config, simulation_metrics)

            # End timer for simulation run.
            end_time = time.time()

            # Add simulation run time to total coordinator time.
            self.coordinator_and_simulation_execution_time += end_time - start_time

            if self.remove_design_point_configuration_dummy_path:
                self.logger.debug("Removing simulation run templates in generated path.")
                fo.remove_design_point_configurations_dummy_path(self.generated_path, pattern=self.remove_design_point_configuration_dummy_path_pattern+"*")
            # self.logger.info("Fitness value: {}".format(fitness_value))
            return fitness_value[0]


    '''
    Set the values of the parameters in the simulation model.

    Args:
        config_values: A list of values for the parameters in the simulation model.
    '''
    def set_param_values(self, config_values):
        raise NotImplementedError("You need to implement: set_param_values().")

    '''
    Set the values of the parameters of multiple agents in the simulation model.

    Args:
        config_values: A list of values for the parameters in the simulation model.
    '''
    def set_agents_param_values(self, config_values_agents):
        raise NotImplementedError("You need to implement: set_agents_param_values().")


    '''
        Collect the simulation stats from the simulation run.

        Args:
            uid: The unique identifier of the simulation run.
    '''
    def obtain_simulation_stats(self, uids):
        raise NotImplementedError("You need to implement: obtain_simulation_stats().")

    def run_single_simulation_configuration(self, step_iteration_data={'step': -2, 'iteration': -2}):
        """
        Runs a single simulation configuration.

        This method creates simulation instances based on the provided configuration and runs the simulation model.
        It also handles the unique identifier (UID) of each simulation instance and aggregates the simulation execution time.

        Returns:
            dict: A dictionary containing the UID of the evaluated simulation instance and its corresponding value.
        """
        start_time = time.time()
        sim_instances = [self.create_dummy(self.config, self.generated_path, uuid.uuid4(), self.inet_path) for _ in range(self._nr_of_sims)]

        uid = sim_instances[0].uid
        if uid in self.uids:
            self.logger.warn(f"UID {uid} already exists in {self.uids}")
        else:
            self.uids.append(uid)

        # Run the configured simulation model.
        self.manager.enqueue_tasks(sim_instances, metadata=step_iteration_data)

        # TEMP: Check uid handling
        evaluated_sim_instances = self.manager.evaluate_all()
        uid = evaluated_sim_instances[0].uid

        end_time = time.time()

        # TODO: Change the determined execution time to obtaining it from the simulation model runtime folder.
        # Aggregate simulation time to a variable of total simulation time.
        self.simulation_execution_time += end_time - start_time

        return {uid: 0}


    def run_multiple_simulation_configuration(self, sim_ids, file_name_fitness_values="fitness_values.json", step_iteration_data={'step': -1, 'iteration': -1}):
        """
        Run multiple simulation configurations.

        Args:
            sim_ids (list): A list of simulation IDs.

        Returns:
            dict: A dictionary mapping Herman's simulation instance UIDs to their corresponding coordinator IDs.
        """
        start_time = time.time()

        # Create/Get queue id for every problem
        queue_name = Path(file_name_fitness_values).stem
        if queue_name not in self.queues:
            self.queues[queue_name] = f'q{len(self.queues)}'
        queue_id = self.queues[queue_name]

        self.logger.debug(f"(Queue: {queue_id}) Enqueing sim instances.")

        # Configuring siminstances.
        sim_instances = [self.create_dummy_parallel(self.config, self.generated_path, sim_id, self.inet_path) for sim_id in sim_ids]
        uids = {sim_instance.uid: id for id, sim_instance in enumerate(sim_instances)}

        # Run the configured simulation model.
        self.manager.enqueue_tasks(sim_instances, queue_id=queue_id, metadata=step_iteration_data)

        self.logger.debug(f"(Queue: {queue_id}) Evaluating simulation instances: {uids} with sim ids: {sim_ids}")

        self.manager.evaluate_queue_all(queue_id)

        end_time = time.time()

        # TODO: Change the determined execution time to obtaining it from the simulation model runtime folder.
        # Aggregate simulation time to a variable of total simulation time.
        self.simulation_execution_time += end_time - start_time

        return uids


    def get_boundaries(self):
        raise NotImplementedError("You need to implement: get_boundaries().")

    def manual_normalization(self):
        raise NotImplementedError("You need to implement: manual_normalization().")

    def determine_boundary_value(self, sim_uid, parameter, boundary):
        raise NotImplementedError("You need to implement: determine_boundary_value().")

    def _check_normalization(self):
        raise NotImplementedError("You need to implement: _check_normalization().")

    def create_dummy(self, *args, **kwargs):
        raise NotImplementedError("You need to implement: create_dummy().")

    def create_dummy_parallel(self, *args, **kwargs):
        raise NotImplementedError("You need to implement: create_dummy_parallel().")

    def remove_sim_instance_output_files(self, sim_uid, csv_file_path, folder_path):
        self.logger.info(f"Removing the output file for simistance {sim_uid}")
        os.remove(csv_file_path)

        for file in self.files_to_keep:
            local_file_path = os.path.join(folder_path, file)
            if os.path.isfile(local_file_path) or os.path.islink(local_file_path):
                os.remove(local_file_path)  # remove the file
            elif os.path.isdir(local_file_path):
                shutil.rmtree(local_file_path)  # remove dir and all contains
            else:
                self.logger.error("During removing sim instance ({}) output files: File {} is not a file or directory. Could not remove in {}.".format(sim_uid, local_file_path, folder_path))

    def update_file_params(self, filepath, new_params):
        directory, filename = os.path.split(filepath)
        temp_filepath = os.path.join(directory, f"temp_{filename}")
        try:
            with open(filepath, 'r') as file_to_read, open(temp_filepath, 'w') as file_to_write:
                for line in file_to_read:
                    updated_line = line
                    for param, value in new_params.items():
                        if line.startswith(param):
                            updated_line = f"{param} = {value}\n"
                            break
                    file_to_write.write(updated_line)

            # Replace original file with updated file
            shutil.move(temp_filepath, filepath)
        except FileNotFoundError:
            print("File not found.")
        except Exception as e:
            print("An error occurred:", e)


    def ignore_files(self, files_to_ignore_patterns):
        import fnmatch  # For wildcard pattern matching

        def _ignore(src, names):
            # src: the directory being visited by copytree()
            # names: a list of its contents (files and subdirectories)
            
            ignored_names = set()
            if not files_to_ignore_patterns:
                return ignored_names # Return empty set if no patterns are provided

            for pattern in files_to_ignore_patterns:
                # fnmatch.filter returns a list of names from 'names' that match 'pattern'
                # Add these to our set of ignored names
                ignored_names.update(fnmatch.filter(names, pattern))
            
            return ignored_names # Return the set of names to ignore in the current directory
        return _ignore


    def duplicate_directory(self, src_dir, dest_dir, files_to_ignore):
        # Delete destination directory if it exists
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        # Duplicate a new custom dummy sim directory and ignore the given filename.
        # files_to_ignore is now a list of patterns (e.g., ["*.txt", "temp_*"])
        shutil.copytree(src_dir,
                        dest_dir,
                        ignore=self.ignore_files(files_to_ignore),              
                        symlinks=True
        )
