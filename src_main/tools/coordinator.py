from hmac import new
from src_main.tools.logger import logger
from src_main.data.collect_data import DataCollector
import src_main.tools.simulation_config.inet_config as inet_config

import os
import shutil
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
import glob
import itertools

from experiments import create_sim_custom_dummy, create_sim_inet_lans_dummy, create_sim_inet_lans_dummy_parallel
from src.manager import Manager
from src.utils.config_reader import Config
from src.utils.config_creator import WorkflowConfig

import uuid


class HeuristicSimulationCoordinator:

    def __del__(self):
        self.manager.shutdown()

    '''
        Initialize the HeuristicSimulationCoordinator.

        Args:
            base_path: The base path of the project.
            coordinator_config_file_path: The path to the configuration file of the coordinator.
            nr_of_agents: The number of agents to run the simulation model with.
    '''
    def __init__(self, base_path, coordinator_config_file_path, nr_of_agents, heuristic_name, nr_of_backbone_switches=6):
        self.base_path = base_path
        self.nr_of_agents = nr_of_agents
        self.nr_of_sims = nr_of_agents
        self.heuristic_name = heuristic_name
        self.nr_of_backbone_switches = nr_of_backbone_switches

        # Predefine the parameters for the execution times.
        self.coordinator_execution_time = 0
        self.simulation_execution_time = 0

        self.uids = []

        # Set up the logger.
        coordinator_log_path = os.path.join(self.base_path, "data/logs/coordinator")
        os.makedirs(coordinator_log_path, exist_ok=True)
        self.logger = logger("coordinator", coordinator_log_path)

        self.logger.info("Reading coordinator config file.")
        self.conf = Config(coordinator_config_file_path, Path(coordinator_log_path), "coordinator_config_manager")

        # Define params to set up the Manager.
        experiments_path = self.conf.tryGet("simulation_model", "simulation_model_paths", "experiments_path")
        sim_model = self.conf.tryGet("simulation_model", "simulation_model_configuration", "sim_model")
        num_nodes = self.conf.tryGet("simulation_model", "simulation_model_configuration", "num_nodes")
        num_workers = self.conf.tryGet("simulation_model", "simulation_model_configuration", "num_workers")
        time_stamp = time.strftime("%Y%m%d_%H%M%S")
        self.data_path = os.path.join(experiments_path, "data", f"campaign_{sim_model}", f"n{str(num_nodes)}_w{str(num_workers)}_s{str(self.nr_of_sims)}", time_stamp)

        # Define params for configuration file creation.
        workflow_config_file = os.path.join(self.data_path, "config.json")
        workflow_results_folder = os.path.join(self.data_path, "results")
        workflow_logs_folder = os.path.join(self.data_path, "logs")
        workflow_runtime_folder = os.path.join(self.data_path, "runtime")

        self.logger.info("Setting up workflow configuration file.")
        self.sims_path = self.conf.tryGet("simulation_model", "simulation_model_paths", "sims_path")
        design_queues = [WorkflowConfig.create_design_point_queue_config("base", 0, "FIFO")]
        cluster_config = self.create_relevant_cluster_config()
        self.workflow_config = WorkflowConfig(self.sims_path, "config", "run_sim", "results", "logs", "out",
                            workflow_results_folder, workflow_logs_folder, workflow_runtime_folder, design_queues, "uuid", cluster_config)
        self.config = self.workflow_config.conf()
        self.workflow_config.write_conf(workflow_config_file)

        self.logger.info("Setting up the Manager.")
        self.manager = Manager(workflow_config_file, workflow_logs_folder)

        # Define variables for coordinator functionalities.

        self.logger.info("Define the base paths for the simulation model instance generation functionalites.")
        self.simulation_model_template_path = self.conf.tryGet("simulation_model", "simulation_model_paths", "simulation_model_template_path")
        self.ini_file_template_path = os.path.join(self.simulation_model_template_path, "largeNet.ini")
        self.generated_simulation_model_path = self.conf.tryGet("simulation_model", "simulation_model_paths", "simulation_model_generated_path")
        self.inet_path = self.conf.tryGet("simulation_model", "simulation_model_paths", "inet_path")
        self.dummy_sim_path = self.conf.tryGet("simulation_model", "simulation_model_paths", "dummy_path")

        self.logger.info("Determine flags for coordinator functionalities.")
        self.store_design_points_metrics_values = self.conf.tryGet("coordinator_functionalities", "store_design_points_metrics_values")
        self.remove_sim_instance_output = self.conf.tryGet("coordinator_functionalities", "remove_sim_instance_output")
        self.remove_sim_instance_configurations = self.conf.tryGet("coordinator_functionalities", "remove_sim_instance_configurations")

        self.logger.info("Define the paths for the simulation model results.")
        self.results_path = self.conf.tryGet("results_path")

        # TODO: Move more functionalities to data collector.
        # TODO: Define config file with a distinct segment for data collector?
        self.logger.info("Define the base paths and variables for simulation instance output functionalities.")
        self.agents_fitness_dir_relative_path = self.conf.tryGet("output_paths", "agents_fitness_values_relative_path")
        self.sim_dummy_directory = self.conf.tryGet("simulation_model", "simulation_model_paths", "dummy_path")
        self.logger.info("Setting up the data collector.")
        dir_design_points_metrics_output = self.conf.tryGet("output_paths", "design_points_metrics_output")
        weight_latency = self.conf.tryGet("fitness_config", "weight_latency")
        weight_cost = self.conf.tryGet("fitness_config", "weight_cost")
        self.data_collector = DataCollector(weight_latency, weight_cost, dir_design_points_metrics_output)

        # TODO: Implement normalisation correctly.
        self.manual_normalization()

        if hasattr(self, "max_datarate"):
            self.logger.info(f"Maximum datarate is: {self.max_datarate}")
            self.logger.info(f"Minimum datarate is: {self.min_datarate}")
        if hasattr(self, "max_cost"):
            self.logger.info(f"Maximum cost is: {self.max_cost}")
            self.logger.info(f"Minimum cost is: {self.min_cost}")


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
        self.duplicate_directory(self.simulation_model_template_path, design_point_path, "largeNet.ini")

        # TODO: Change the hardcoded ini filename to a variable in the configuration file.
        # Write an updated version of the ignored param value file from the directory that was just duplicated.
        template_ini_file_path = os.path.join(self.simulation_model_template_path, "largeNet.ini")
        design_point_ini_file_path = os.path.join(design_point_path, "largeNet.ini")

        # TODO: Change the hardcoded number of backbone switches to a variable.
        # Write a new ini file with the parameter configurations.
        inet_config.generate_inet_lans_config(
            template_ini_file_path,
            design_point_ini_file_path,
            agent_configuration,
            self.nr_of_backbone_switches
        )


    def locally_store_agents_fitness_values(self, fitness_values):
        """
        Locally stores the fitness values of agents in a JSON file.

        Args:
            fitness_values (dict): A dictionary containing the fitness values of agents.

        Returns:
            None
        """
        self.logger.info(f"Storing fitness values locally at {self.agents_fitness_dir_relative_path}")
        agents_fitness_dir = os.path.join(self.base_path, self.agents_fitness_dir_relative_path)
        os.makedirs(agents_fitness_dir, exist_ok=True)
        fitness_values_file_path = os.path.join(agents_fitness_dir, "fitness_values.json")
        if os.path.exists(fitness_values_file_path):
            os.remove(fitness_values_file_path)
        with open(fitness_values_file_path, 'w') as f:
            json.dump(fitness_values, f)


    '''
        Run the simulation model with the given configuration values.

        Args:
            fitfunc: The fitness function to evaluate the simulation run.
            config_values: A list of values for the parameters in the simulation model.

        Returns:
            The fitness value of the simulation run.
    '''
    def simulation_run(self, fitfunc, config_values):

        # Start timer for simulation run.
        start_time = time.time()
        # TODO: Merge the following two if statements into one. -> CUSTOMHys should be able to handle both situations?
        if self.nr_of_agents > 1:
            self.logger.info(f"Determining the simulation model parameters for the configuration values.")
            configurations = self.set_agents_param_values(config_values)
            sim_ids = []
            self.logger.info("Generating sim instances for the agents.")
            for agent_configuration in configurations:
                sim_id = uuid.uuid4()
                sim_ids.append(sim_id)
                self.logger.debug(f"Generating simulation instance for design point: {sim_id}.")
                self.generate_design_point(sim_id, agent_configuration)

            self.logger.info("Run the generated simulation instances.")
            uids = self.run_multiple_simulation_configuration(sim_ids)

            self.logger.info("Collect the simulation stats from the simulation instances runs.")
            simulation_metrics = self.obtain_simulation_stats(uids)

            self.logger.info("Locally storing the agents fitness values.")
            fitness_config = self.conf.tryGet("fitness_config")
            fitness_values = fitfunc(fitness_config, simulation_metrics)
            self.locally_store_agents_fitness_values(fitness_values)

            # End timer for simulation run.
            end_time = time.time()

            # Add simulation run time to total coordinator time.
            self.coordinator_execution_time += end_time - start_time

            sim_ids.clear()
            if self.remove_sim_instance_configurations:
                self.remove_simulation_instance_configurations()

            return 0

        else:
            # Set the values of the parameters in the simulation model.
            configurations = self.set_param_values(config_values)

            sim_id = uuid.uuid4()
            self.generate_design_point(sim_id, configurations)

            # Run the simulation model.
            uid = self.run_single_simulation_configuration()

            # Collect the simulation stats from the simulation run.
            simulation_metrics = self.obtain_simulation_stats(uid)

            fitness_config = self.conf.tryGet("fitness_config")
            fitness_value = fitfunc(fitness_config, simulation_metrics)

            # End timer for simulation run.
            end_time = time.time()

            # Add simulation run time to total coordinator time.
            self.coordinator_execution_time += end_time - start_time

            if self.remove_sim_instance_configurations:
                self.remove_simulation_instance_configurations()

            # self.logger.info("Fitness value: {}".format(fitness_value))
            return fitness_value[0]


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

        for param in params:
            config_pattern = param["configuration_pattern"]
            param_values = param["values"]

            # Determine the param value index by segment indexing the config values from CUSTOMHys.
            num_segments = len(param_values)
            value_indices = np.floor(config_values * num_segments).astype(int)
            value_indices[config_values == 1.0] = num_segments - 1

            # TODO: Optimize this operations as it now parses and has duplicate code.
            # Create a configuration for each parameter.
            for switch_idx, value_idx in enumerate(value_indices):
                selected_param_value = param_values[value_idx]

                # Conditional handling for the switch gates where the first switch is handled differently.
                first_switch_gate_index = "1" if switch_idx > 0 else "0"
                second_switch_gate_index = "0"

                # Replace placeholders in the configuration pattern are replaced with the corresponding indices.
                current_param_pattern = config_pattern.replace("__switch_index", str(switch_idx))
                current_param_pattern = current_param_pattern.replace("__gate_index", first_switch_gate_index)

                # Create a configuration for the current parameter.
                configurations.append({
                    "config_pattern": current_param_pattern,
                    "value": selected_param_value
                })

                # Replace placeholders in the configuration pattern are replaced with the corresponding indices.
                current_param_pattern = config_pattern.replace("__switch_index", str(switch_idx+1))
                current_param_pattern = current_param_pattern.replace("__gate_index", second_switch_gate_index)

                configurations.append({
                    "config_pattern": current_param_pattern,
                    "value": selected_param_value
                })

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
                for switch_idx, value_idx in enumerate(value_indices):
                    selected_param_value = param_values[value_idx]
                    # Conditional handling for the switch gates where the first switch is handled differently.
                    first_switch_gate_index = "1" if switch_idx > 0 else "0"
                    second_switch_gate_index = "0"

                    # Replace placeholders in the configuration pattern are replaced with the corresponding indices.
                    current_param_pattern = config_pattern.replace("__switch_index", str(switch_idx))
                    current_param_pattern = current_param_pattern.replace("__gate_index", first_switch_gate_index)

                    # Create a configuration for the current parameter.
                    agent_configuration.append({
                        "config_pattern": current_param_pattern,
                        "value": selected_param_value
                    })

                    # Replace placeholders in the configuration pattern are replaced with the corresponding indices.
                    current_param_pattern = config_pattern.replace("__switch_index", str(switch_idx+1))
                    current_param_pattern = current_param_pattern.replace("__gate_index", second_switch_gate_index)

                    agent_configuration.append({
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
    def obtain_simulation_stats(self, uids):

        # TODO: Should rename fitness_values variable accross the board to something else, because this is not the correct term.
        # Load the transformed outputted data from the simulation run.
        fitness_values = []

        for sim_uid, agent_id in uids.items():

            # TODO: Change scavetool output filename to something more descriptive.
            csv_file_path = os.path.join(self.data_path, "results", sim_uid, "x.csv")
            df = pd.read_csv(csv_file_path)

            # Determine end-to-end delay statistics.
            latency_df = df[df["type"] == "statistic"]
            latency_df = latency_df[latency_df["module"].str.endswith(".cli")]
            latency_df = latency_df[latency_df["name"].str.startswith("endToEndDelay")]

            # Get the cost of all components with a cost paramater in the simulation model.
            cost_df = df[df['type'] == "param"]
            cost_df = cost_df[cost_df["name"] == "cost"]

            # Store design point metrics output.
            if self.store_design_points_metrics_values:
                self.logger.info(f"Storing metrics output values for siminstance {sim_uid}.")
                self.data_collector.store_design_point_metrics(latency_df, cost_df, sim_uid, self.heuristic_name)

            # Remove the csv file.
            if self.remove_sim_instance_output:
                self.logger.info(f"Removing the output file for simistance {sim_uid}")
                os.remove(csv_file_path)

            fitness_values.append({
                agent_id: {
                    "latency": latency_df['mean'].mean(),
                    "network_cost": cost_df['value'].astype(float).sum()
                }
            })

        # return the average of the column 'mean' in the dataframe.
        return fitness_values


    def run_single_simulation_configuration(self):
        """
        Runs a single simulation configuration.

        This method creates simulation instances based on the provided configuration and runs the simulation model.
        It also handles the unique identifier (UID) of each simulation instance and aggregates the simulation execution time.

        Returns:
            dict: A dictionary containing the UID of the evaluated simulation instance and its corresponding value.
        """
        start_time = time.time()
        sim_instances = [create_sim_inet_lans_dummy(self.config, self.dummy_sim_path, uuid.uuid4(), self.inet_path) for _ in range(self.nr_of_sims)]

        uid = sim_instances[0].uid
        if uid in self.uids:
            self.logger.warn(f"UID {uid} already exists in {self.uids}")
        else:
            self.uids.append(uid)

        # Run the configured simulation model.
        self.manager.enqueue_tasks(sim_instances)

        # TEMP: Check uid handling
        evaluated_sim_instances = self.manager.evaluate_all()
        uid = evaluated_sim_instances[0].uid

        end_time = time.time()

        # TODO: Change the determined execution time to obtaining it from the simulation model runtime folder.
        # Aggregate simulation time to a variable of total simulation time.
        self.simulation_execution_time += end_time - start_time

        return {uid: 0}


    def run_multiple_simulation_configuration(self, sim_ids):
        """
        Run multiple simulation configurations.

        Args:
            sim_ids (list): A list of simulation IDs.

        Returns:
            dict: A dictionary mapping Herman's simulation instance UIDs to their corresponding coordinator IDs.
        """
        start_time = time.time()

        # Configuring siminstances.
        sim_instances = [create_sim_inet_lans_dummy_parallel(self.config, self.dummy_sim_path, sim_id, self.inet_path) for sim_id in sim_ids]
        uids = {sim_instance.uid: id for id, sim_instance in enumerate(sim_instances)}

        self.logger.debug(f"Enqueing sim instances.")

        # Run the configured simulation model.
        self.manager.enqueue_tasks(sim_instances)

        self.logger.debug(f"Evaluating simulation instances: {uids} with sim ids: {sim_ids}")

        self.manager.evaluate_all()

        end_time = time.time()

        # TODO: Change the determined execution time to obtaining it from the simulation model runtime folder.
        # Aggregate simulation time to a variable of total simulation time.
        self.simulation_execution_time += end_time - start_time

        return uids


    def remove_simulation_instance_configurations(self):
        """
        Removes simulation run templates from the dummy path and sims path.

        This method removes all the simulation run templates that match the pattern
        "custom_dummy_*" from the dummy path and also removes the entire sims path.
        After removing the templates and sims path, it recreates the sims path.

        Args:
            None

        Returns:
            None
        """
        self.logger.info("Removing simulation run templates in dummy path.")
        pattern = "custom_dummy_*"
        simulation_run_dir_pattern = os.path.join(self.sim_dummy_directory, pattern)
        matching_sim_dummy_dir = [path for path in glob.glob(simulation_run_dir_pattern) if os.path.isdir(path)]
        for directory in matching_sim_dummy_dir:
            shutil.rmtree(directory)

        # TODO: Should I make a distinct function of the removal from the sims path?
        # self.logger.info("Removing simulation run templates in sims path.")
        # shutil.rmtree(self.sims_path)
        # os.makedirs(self.sims_path)


    def get_datarate_cost_boundaries(self):
        return self.min_datarate, self.max_datarate, self.min_cost, self.max_cost


    # TODO: Remove redundancy of multiple ini file writing.
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
        simulation_model_params = self.conf.tryGet("simulation_model", "simulation_model_params")
        boundaries = {}
        for simulation_model_param in simulation_model_params:
            parameter_name = simulation_model_param['param_name']
            if parameter_name != "cable_colour":
                boundaries[simulation_model_param['param_name']] = {
                    "max": simulation_model_param['values'][-1],
                    "min": simulation_model_param['values'][0]
                }

        # Generate simulation instances the min and max parameter value configurations.
        sim_ids = []
        for parameter, boundaries in boundaries.items():
            for boundary in boundaries:
                value = boundaries[boundary]
                self.logger.debug(f"Generating normalization sim instance for parameter: {parameter} - boundary: {boundary} - value: {value}")
                sim_id = uuid.uuid4()
                configuration = [
                    {
                        "config_pattern": f"**.switchBB[0].ethg$o[0].channel.{parameter}",
                        "value": value
                    },
                    {
                        "config_pattern": f"**.switchBB[1..{self.nr_of_backbone_switches-2}].ethg$o[0..1].channel.{parameter}",
                        "value": value
                    },
                    {
                        "config_pattern": f"**.switchBB[{self.nr_of_backbone_switches-1}].ethg$o[0].channel.{parameter}",
                        "value": value
                    }
                ]
                self.generate_design_point(sim_id, configuration)
                sim_ids.append(sim_id)

                self.logger.info("Evaluating the generated simulation instances.")
                uids = self.run_multiple_simulation_configuration(sim_ids)

                self.logger.info("Determine the boundary value for the objective.")
                for sim_uid in uids.keys():
                    self.determine_boundary_value(sim_uid, parameter, boundary)

                sim_ids.clear()

        if self.remove_sim_instance_configurations:
            self.remove_simulation_instance_configurations()


    def determine_boundary_value(self, sim_uid, parameter, boundary):
        csv_file_path = os.path.join(self.data_path, "results", sim_uid, "x.csv")
        df = pd.read_csv(csv_file_path)

        # Determine boundary values for the datarate parameter.
        if parameter == "datarate":
            cli_df = df[df["module"].fillna("").str.endswith(".cli")]
            latency_df = cli_df[(cli_df["type"] == "statistic") & (cli_df["name"].str.startswith("endToEndDelay"))]
            mean_end_to_end_delay = latency_df['mean']
            _ = latency_df['stddev']
            _ = latency_df['min']
            _ = latency_df['max']
            if boundary == "min":
                self.min_datarate = mean_end_to_end_delay.to_numpy()[0]
            elif boundary == "max":
                self.max_datarate = mean_end_to_end_delay.to_numpy()[0]
        # Deterine boundary values for the cost parameter.
        elif parameter == "cost":
            cost_df = df[df['type'] == "param"]
            cost_df = cost_df[cost_df["name"] == "cost"]
            if boundary == "min":
                self.min_cost = cost_df['value'].astype(float).sum()
            elif boundary == "max":
                self.max_cost = cost_df['value'].astype(float).sum()


    '''
        Perform the parameter tuning workflow.
    '''
    def parameter_tuning_workflow(self):
        self.logger.info("Configuring parameter tuning workflow..")
        parameter_names = self.conf.tryGet("parameter_tuning", "parameter_names")
        parameter_ranges = self.conf.tryGet("parameter_tuning", "parameter_ranges")
        parameter_tuning_path = os.path.join(self.results_path, "parameter_tuning")
        os.makedirs(parameter_tuning_path, exist_ok=True)

        sim_ids = []
        for idx, parameter_name in enumerate(parameter_names):
            parameter_range = parameter_ranges[idx]
            self.logger.info(f"Tuning parameter {parameter_name} within range {parameter_range}.")
            for parameter_value in parameter_range:
                sim_id = uuid.uuid4()
                sim_ids.append(sim_id)
                sim_instance_path = self.generated_simulation_model_path + f"_{sim_id}"

                # Duplicate the preferred dummy_sim directory to the sim_instance directory.
                self.duplicate_directory(self.simulation_model_template_path, sim_instance_path, file_to_ignore="largeNet.ini")
                # Define the path to the sim_instance ini file.
                sim_instance_ini_file_path = os.path.join(sim_instance_path, "largeNet.ini")

                # Write the sim_instance ini file with the provided parameter configurations.
                self.write_sim_instance_ini_file(
                    self.ini_file_template_path,
                    sim_instance_ini_file_path,
                    {parameter_name: parameter_value},
                    self.nr_of_backbone_switches
                )

        self.logger.info("Evaluating the generated simulation instances.")
        uids = self.run_multiple_simulation_configuration(sim_ids)

        self.logger.info("Saving the parameter tuning results for the simulation instances.")
        for sim_uid in uids.keys():
            self.determine_sim_instance_parameter_tuning_results(sim_uid, parameter_names)

        sim_ids.clear()
        if self.remove_sim_instance_configurations:
            self.remove_simulation_instance_configurations()


    '''
        Determine the parameter tuning results for the simulation run.

        Args:
            sim_uid: The unique identifier of the simulation run.
            parameter_names: The names of the parameters tuned in the simulation model.
    '''
    def determine_sim_instance_parameter_tuning_results(self, sim_uid, parameter_names):
        # TODO: Change scavetool output filename to something more descriptive.
        csv_file_path = os.path.join(self.data_path, "results", sim_uid, "x.csv")
        df = pd.read_csv(csv_file_path)

        # TODO: Make the following code more generic and less hardcoded.
        cli_df = df[df["module"].fillna("").str.endswith(".cli")]
        latency_df = cli_df[(cli_df["type"] == "statistic") & (cli_df["name"].str.startswith("endToEndDelay"))]
        packet_df = cli_df[(cli_df['type'] == "scalar") & (cli_df['name'].str.startswith("packet"))]

        # Remove the csv file.
        if self.remove_sim_instance_output:
            self.logger.info(f"Removing the output file for simistance {sim_uid}")
            os.remove(csv_file_path)

        self.save_parameter_tuning_results(latency_df, packet_df, sim_uid, parameter_names)


    '''
        Save the parameter tuning results from the simulation run.

        Args:
            latency_df: The dataframe containing the latency statistics of the simulation run.
            packet_df: The dataframe containing the packet statistics of the simulation run.
            sim_uid: The unique identifier of the simulation run.
            parameter_names: The names of the parameters tuned in the simulation model.
    '''
    # TODO: Move to data module.
    def save_parameter_tuning_results(self, latency_df, packet_df, sim_uid, parameter_names):

        # Determine the parameter tuning results.
        count_packets_sent = packet_df[(packet_df['name'] == "packetSent:count")]['value'].values.sum()
        packets_sent_sum = packet_df[(packet_df['name'] == "packetSent:sum(packetBytes)")]['value'].values.sum()
        count_packets_received = packet_df[(packet_df['name'] == "packetReceived:count")]['value'].values.sum()
        packets_received_sum = packet_df[(packet_df['name'] == "packetReceived:sum(packetBytes)")]['value'].values.sum()
        mean_end_to_end_delay = latency_df['mean'].values.mean()
        stddev_end_to_end_delay = latency_df['stddev'].values.mean()
        min_end_to_end_delay = latency_df['min'].values.mean()
        max_end_to_end_delay = latency_df['max'].values.mean()

        # Append the parameter tuning results to the parameter tuning storage.
        parameter_tune_run_file_name = os.path.join(self.results_path, "parameter_tuning", "_".join(parameter_names) + ".csv")
        append_df = pd.DataFrame([[sim_uid, count_packets_sent, packets_sent_sum, count_packets_received, packets_received_sum, mean_end_to_end_delay, stddev_end_to_end_delay, min_end_to_end_delay, max_end_to_end_delay]],
                                columns=['SimulationID', 'CountPacketsSent', 'PacketsSentSum', 'CountPacketsReceived', 'PacketsReceivedSum', 'MeanEndToEndDelay', 'StddevEndToEndDelay', 'MinEndToEndDelay', 'MaxEndToEndDelay'])
        append_df.to_csv(parameter_tune_run_file_name, mode='a', header=not os.path.exists(parameter_tune_run_file_name), index=False)


    def determine_design_space(self):
        """
        Determines the design space for the simulation model.

        This method generates all possible combinations of cable options and evaluates each design point
        by running multiple simulation configurations. It then determines the design point metrics based on
        the simulation results.

        Returns:
            None
        """
        self.logger.info("Determining the design space for the simulation model.")
        cable_options = self.conf.tryGet("cable_options")
        sim_ids = []
        for permutation in itertools.product(cable_options, repeat=self.nr_of_backbone_switches):
            sim_id = uuid.uuid4()
            sim_ids.append(sim_id)
            sim_instance_path = self.generated_simulation_model_path + f"_{sim_id}"

            # Duplicate the preferred dummy_sim directory to the sim_instance directory.
            self.duplicate_directory(self.simulation_model_template_path, sim_instance_path, file_to_ignore="largeNet.ini")
            # Define the path to the sim_instance ini file.
            sim_instance_ini_file_path = os.path.join(sim_instance_path, "largeNet.ini")

            # Write the sim_instance ini file with the provided cable options permutation.
            self.write_permutation_ini_file(
                self.ini_file_template_path,
                sim_instance_ini_file_path,
                permutation
            )

        self.logger.info("Evaluating every possible design point.")
        uids = self.run_multiple_simulation_configuration(sim_ids)

        self.determine_design_point_metrics(uids)

        sim_ids.clear()
        if self.remove_sim_instance_configurations:
            self.remove_simulation_instance_configurations()


    def determine_design_point_metrics(self, uids):
        """
        Determines the design point metrics for every evaluated design point.

        Args:
            uids (dict): A dictionary containing the unique identifiers for each design point.

        Returns:
            None
        """
        self.logger.info("Determining the design point metrics for every evaluated design point.")
        for sim_uid in uids.keys():
            # TODO: Change scavetool output filename to something more descriptive.
            csv_file_path = os.path.join(self.data_path, "results", sim_uid, "x.csv")
            df = pd.read_csv(csv_file_path)

            # Determine end-to-end delay statistics.
            latency_df = df[df["type"] == "statistic"]
            latency_df = latency_df[latency_df["module"].str.endswith(".cli")]
            latency_df = latency_df[latency_df["name"].str.startswith("endToEndDelay")]

            # Get the cost of all components with a cost parameter in the simulation model.
            cost_df = df[df['type'] == "param"]
            cost_df = cost_df[cost_df["name"] == "cost"]

            # Store design point metrics output.
            self.data_collector.store_design_point_metrics(latency_df, cost_df, sim_uid, self.heuristic_name)


    @staticmethod
    def process_simulation_output(sim_runtime_csv_file_path, column_name):
        df = pd.read_csv(sim_runtime_csv_file_path)
        sim_exec_time = df.loc[0, column_name]
        return sim_exec_time


    @staticmethod
    def ignore_file(file_name):
        def _ignore(_, filenames):
            return [name for name in filenames if name == file_name]
        return _ignore


    @staticmethod
    def write_permutation_ini_file(ini_file_template_path, design_point_ini_file_path, permutation):
        # Step 1: Read the contents of the template INI file
        with open(ini_file_template_path, 'r') as template_file:
            template_contents = template_file.read()

        # Step 2: Open the design point INI file for writing
        with open(design_point_ini_file_path, 'w') as design_file:
            # Write the template contents to the design point INI file
            design_file.write(template_contents)

            # Step 3: Write additional information for each permutation
            for switch_idx, cable_option in enumerate(permutation):
                cable_option_rate = cable_option['datarate']
                cable_option_colour = cable_option['cable_colour']
                if switch_idx == 0:
                    design_file.write(f"**.switchBB[0].ethg$o[0].channel.datarate = {cable_option_rate}\n")
                    design_file.write(f"**.switchBB[1].ethg$o[0].channel.datarate = {cable_option_rate}\n")
                    design_file.write(f"**.switchBB[0].ethg$o[0].channel.display-string = ls={cable_option_colour},3,s;\n")
                    design_file.write(f"**.switchBB[1].ethg$o[0].channel.display-string = ls={cable_option_colour},3,s;\n")
                else:
                    design_file.write(f"**.switchBB[{switch_idx}].ethg$o[1].channel.datarate = {cable_option_rate}\n")
                    design_file.write(f"**.switchBB[{switch_idx+1}].ethg$o[0].channel.datarate = {cable_option_rate}\n")
                    design_file.write(f"**.switchBB[{switch_idx}].ethg$o[1].channel.display-string = ls={cable_option_colour},3,s;\n")
                    design_file.write(f"**.switchBB[{switch_idx+1}].ethg$o[0].channel.display-string = ls={cable_option_colour},3,s;\n")
                design_file.write(f"**.switchBB[{switch_idx+1}].ethg$o[0].channel.cost = {cable_option['cost']}\n")


    @staticmethod
    def write_sim_instance_ini_file(template_ini_file_path, sim_instance_ini_file_path, configurations, nr_of_backbone_switches):
        """
        Write a simulation instance INI file based on a template INI file and given configurations.
        Parameters:
        - template_ini_file_path (str): The path to the template INI file.
        - sim_instance_ini_file_path (str): The path to the simulation instance INI file to be created.
        - configurations (dict): A dictionary containing the configuration patterns and their corresponding values.
        - nr_of_backbone_switches (int): The number of switches on the backbone.
        Returns:
        None
        """
        configuration_parameters = configurations.keys()
        with open(template_ini_file_path, 'r') as template_ini_file, open(sim_instance_ini_file_path, 'w') as sim_instance_ini_file:
            for line in template_ini_file:
                line_written = False
                for configuration_parameter in configuration_parameters:

                    # Update the default parameter values with the given configuration values.
                    if line.startswith(configuration_parameter):
                        configuration_value = configurations[configuration_parameter]
                        sim_instance_ini_file.write(f"{configuration_parameter} = {configuration_value}\n")
                        line_written = True
                        break

                    # Update the number of backbone switches accordingly.
                    elif line.startswith("LargeNet.n =") and configuration_parameter == "datarate":
                        sim_instance_ini_file.write(f"LargeNet.n = {nr_of_backbone_switches}   # number of switches on backbone")
                        line_written = True
                        break

                    # Update the cable rate parameter values with the given configuration values.
                    elif line.startswith("# Parameter tuning for cable rate patterns below this line.") and configuration_parameter == "datarate":
                        sim_instance_ini_file.write(line)

                        # Write the cable rate configurations for each backbone switch with the given configuration values.
                        datarate = configurations[configuration_parameter]['rate']
                        sim_instance_ini_file.write(f"**.switchBB[0].ethg$o[0].channel.datarate = {datarate}\n")
                        sim_instance_ini_file.write(f"**.switchBB[1..{nr_of_backbone_switches-2}].ethg$o[0..1].channel.datarate = {datarate}\n")
                        sim_instance_ini_file.write(f"**.switchBB[{nr_of_backbone_switches-1}].ethg$o[0].channel.datarate = {datarate}\n")

                        # Write the cable colour configurations for each backbone switch with the given colour.
                        cable_colour = configurations[configuration_parameter]['colour']
                        sim_instance_ini_file.write(f"**.switchBB[0].ethg$o[0].channel.display-string = ls={cable_colour},3,s;\n")
                        sim_instance_ini_file.write(f"**.switchBB[1..{nr_of_backbone_switches-2}].ethg$o[0..1].channel.display-string = ls={cable_colour},3,s;\n")
                        sim_instance_ini_file.write(f"**.switchBB[{nr_of_backbone_switches-1}].ethg$o[0].channel.display-string = ls={cable_colour},3,s;\n")
                        line_written = True
                        break
                if not line_written:
                    sim_instance_ini_file.write(line)


    @staticmethod
    def update_file_params(filepath, new_params):
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


    @staticmethod
    def duplicate_directory(src_dir, dest_dir, file_to_ignore):
        # Delete destination directory if it exists
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        # Duplicate a new custom dummy sim directory and ignore the given filename.
        shutil.copytree(src_dir,
                        dest_dir,
                        ignore=HeuristicSimulationCoordinator.ignore_file(file_to_ignore)
        )
