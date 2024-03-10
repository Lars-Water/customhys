from src_main.tools.logger import logger

import os
import shutil
import pandas as pd
import numpy as np
import subprocess
import time
import json
from pathlib import Path
import glob

from concurrent.futures import ThreadPoolExecutor

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
            parameter_tuning: A flag to indicate whether the coordinator should perform parameter tuning or not.
    '''
    def __init__(self, base_path, coordinator_config_file_path, nr_of_agents, parameter_tuning=False):
        self.base_path = base_path
        self.nr_of_agents = nr_of_agents
        self.nr_of_sims = nr_of_agents

        # Predefine the parameters for the execution times.
        self.coordinator_execution_time = 0
        self.simulation_execution_time = 0

        self.uids = []

        # # Set up the logger.
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
        threads_per_worker = self.conf.tryGet("simulation_model", "simulation_model_configuration", "threads_per_worker")
        time_stamp = time.strftime("%Y%m%d_%H%M%S")
        self.data_path = os.path.join(experiments_path, "data", f"campaign_{sim_model}", f"n{str(num_nodes)}_w{str(num_workers)}_s{str(self.nr_of_sims)}", time_stamp)

        # Define params for configuration file creation.
        workflow_config_file = os.path.join(self.data_path, "config.json")
        workflow_results_folder = os.path.join(self.data_path, "results")
        workflow_logs_folder = os.path.join(self.data_path, "logs")
        workflow_runtime_folder = os.path.join(self.data_path, "runtime")

        self.logger.info("Setting up workflow configuration file.")
        sims_path = self.conf.tryGet("simulation_model", "simulation_model_paths", "sims_path")
        design_queues = [WorkflowConfig.create_design_point_queue_config("base", 0, "FIFO")]
        cluster_config = self.create_relevant_cluster_config()
        self.workflow_config = WorkflowConfig(sims_path, "config", "run_sim", "results", "logs", "out",
                            workflow_results_folder, workflow_logs_folder, workflow_runtime_folder, design_queues, "uuid", cluster_config)
        self.config = self.workflow_config.conf()
        self.workflow_config.write_conf(workflow_config_file)

        self.logger.info("Setting up the Manager.")
        self.manager = Manager(workflow_config_file, workflow_logs_folder)

        # Define variables for coordinator functionalities.

        self.logger.info("Determining the paths for the simulation model.")
        self.simulation_model_template_path = self.conf.tryGet("simulation_model", "simulation_model_paths", "simulation_model_template_path")
        self.ini_file_template_path = os.path.join(self.simulation_model_template_path, "largeNet.ini")
        self.generated_simulation_model_path = self.conf.tryGet("simulation_model", "simulation_model_paths", "simulation_model_generated_path")

        self.logger.info("Determine flags for coordinator functionalities.")
        self.store_design_points_metrics_values = self.conf.tryGet("coordinator_functionalities", "store_design_points_metrics_values")
        self.remove_sim_instance_output = self.conf.tryGet("coordinator_functionalities", "remove_sim_instance_output")

        self.logger.info("Determine the paths for the simulation model results.")
        self.results_path = self.conf.tryGet("results_path")
        if parameter_tuning:
            # TODO: Change the hardcoded filename to a variable in the configuration file.
            parameter_tuning_path = os.path.join(self.results_path, "parameter_tuning")
            os.makedirs(parameter_tuning_path, exist_ok=True)
            self.parameter_tuning_file = os.path.join(parameter_tuning_path, "parameter_tuning_results.csv")


    '''
        Run the simulation model with the given configuration values.

        Args:
            fitfunc: The fitness function to evaluate the simulation run.
            config_values: A list of values for the parameters in the simulation model.

        Returns:
            The fitness value of the simulation run.
    '''
    def simulation_run(self, fitfunc, config_values):
        # self.logger.info(f"Simulation run with config values:\n{config_values}")

        # Start timer for simulation run.
        start_time = time.time()
        # TODO: Merge the following two if statements into one. -> CUSTOMHys should be able to handle both situations?
        if self.nr_of_agents > 1:
            # Set the values of the parameters in the simulation model.
            configurations = self.set_agents_param_values(config_values)
            sim_ids = []
            for agent_configuration in configurations:
                sim_id = uuid.uuid4()
                sim_ids.append(sim_id)
                src_dir = self.conf.tryGet("simulation_model", "simulation_model_paths", "simulation_model_template_path")
                dest_dir = self.conf.tryGet("simulation_model", "simulation_model_paths", "simulation_model_generated_path") + f"_{sim_id}"

                # TODO: Change the hardcoded ini filename to a variable in the configuration file.
                # Duplicate the preferred dummy_sim directory to the custom directory.
                self.duplicate_directory(src_dir, dest_dir, "largeNet.ini")

                # TODO: Change the hardcoded ini filename to a variable in the configuration file.
                # Write an updated version of the ignored param value file from the directory that was just duplicated.
                old_ini_file_path = os.path.join(src_dir, "largeNet.ini")
                new_ini_file_path = os.path.join(dest_dir, "largeNet.ini")

                # TODO: Change the hardcoded number of backbone switches to a variable.
                # Write a new ini file with the parameter configurations.
                self.write_new_ini_file_new(
                    old_ini_file_path,
                    new_ini_file_path,
                    agent_configuration,
                    config_values.shape[1] + 1
                )

            # Run the simulation model.
            uids = self.run_multiple_simulation_configuration(sim_ids)

            # Collect the simulation stats from the simulation run.
            simulation_metrics = self.obtain_simulation_stats(uids)

            fitness_config = self.conf.tryGet("fitness_config")
            fitness_values = fitfunc(fitness_config, simulation_metrics)

            # TODO: Make distinct function for writing fitness values to a json file.
            # Write fitness values to a json file. Remove the file if it already exists first.
            agents_fitness_dir_relative_path = self.conf.tryGet("output_paths", "agents_fitness_values_relative_path")
            agents_fitness_dir = os.path.join(self.base_path, agents_fitness_dir_relative_path)
            os.makedirs(agents_fitness_dir, exist_ok=True)
            fitness_values_file_path = os.path.join(agents_fitness_dir, "fitness_values.json")
            # Remove the file if it already exists.
            if os.path.exists(fitness_values_file_path):
                os.remove(fitness_values_file_path)
            with open(fitness_values_file_path, 'w') as f:
                json.dump(fitness_values, f)

            # End timer for simulation run.
            end_time = time.time()

            # Add simulation run time to total coordinator time.
            self.coordinator_execution_time += end_time - start_time

            sim_ids.clear()
            self.remove_simulation_run_dirs()

            return 0

        else:
            # Set the values of the parameters in the simulation model.
            configurations = self.set_param_values(config_values)

            src_dir = self.conf.tryGet("simulation_model", "simulation_model_paths", "simulation_model_template_path")
            dest_dir = self.conf.tryGet("simulation_model", "simulation_model_paths", "simulation_model_generated_path")

            # TODO: Change the hardcoded ini filename to a variable in the configuration file.
            # Duplicate the preferred dummy_sim directory to the custom directory.
            self.duplicate_directory(src_dir, dest_dir, "largeNet.ini")

            # TODO: Change the hardcoded ini filename to a variable in the configuration file.
            # Write an updated version of the ignored param value file from the directory that was just duplicated.
            old_ini_file_path = os.path.join(src_dir, "largeNet.ini")
            new_ini_file_path = os.path.join(dest_dir, "largeNet.ini")

            # TODO: Change the hardcoded number of backbone switches to a variable.
            # Write a new ini file with the parameter configurations.
            self.write_new_ini_file_new(
                old_ini_file_path,
                new_ini_file_path,
                configurations,
                config_values.size + 1
            )

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

            # self.logger.info("Fitness value: {}".format(fitness_value))
            return fitness_value[0]


    '''
        Shutdown the Manager.
    '''
    def shutdown_manager(self):
        self.manager.shutdown()


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
            num_configs = param["number_of_configs"]
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
        self.logger.info(f"Determining the simulation model parameters for the configuration values...")

        params = self.conf.tryGet("simulation_model", "simulation_model_params")
        configurations = []

        for config_values_agent in config_values_agents:
            agent_configuration = []

            for param in params:
                config_pattern = param["configuration_pattern"]
                param_values = param["values"]

                # Determine the param value index by segment indexing the config values from CUSTOMHys.
                num_segments = len(param_values)
                value_indices = np.floor(config_values_agent * num_segments).astype(int)
                value_indices[config_values_agent == 1.0] = num_segments - 1

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
            configurations.append(agent_configuration)

        return configurations


    # TODO: Remove this function? -> Moved to siminstance.py
    # '''
    #     Transform the scalar files generated by the simulation model into a csv format.
    # '''
    # def transform_scalar_files(self, uids):
    #     scavetool_command = "opp_scavetool export -F CSV-R -o x.csv *.sca"

    #     for sim_uid in uids.keys():
    #         sim_results_directory = os.path.join(self.data_path, "results", sim_uid)

    #         # Attempt to transform the scalar files in the given results directory.
    #         self.logger.info(f"Replacing scalar files by csv through transformation and removal in directory: {sim_results_directory}")
    #         try:
    #             subprocess.run(scavetool_command, shell=True, cwd=sim_results_directory, check=True)
    #         except subprocess.CalledProcessError as e:
    #             self.logger.error(f"Error executing the command: {e}")

    #         # Remove the scalar files in the given results directory.
    #         for file in os.listdir(sim_results_directory):
    #             if file.endswith(".sca"):
    #                 os.remove(os.path.join(sim_results_directory, file))


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
            self.logger.info(f"Determining simulation statistics for siminstance {sim_uid}.")

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
                self.store_design_point_metrics(latency_df, cost_df, sim_uid)

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


    '''
        Evaluate the fitness value of the simulation run.

        Args:
            simulation_metrics: The simulation metrics obtained from the simulation run.
    '''
    def fitness_evaluation(self, simulation_metrics):
        fitness_config = self.conf.tryGet("fitness_config")
        fitness_function = fitness_config["fitness_function"]

        # Fitness value evaluates the objectives for latency and network cost.
        if fitness_function == "latency_cost":
            weight_latency = fitness_config["weight_latency"]
            weight_cost = fitness_config["weight_cost"]
            fitness_value = (weight_latency * (1 / simulation_metrics["latency"])) + (weight_cost * simulation_metrics["network_cost"])
            return fitness_value

        # TODO: Add other fitness functions here.
        else:
            return 0


    def run_single_simulation_configuration(self):
        # Configure siminstances.
        inet_path = self.conf.tryGet("simulation_model", "simulation_model_paths", "inet_path")
        dummy_sim_path = self.conf.tryGet("simulation_model", "simulation_model_paths", "dummy_path")

        start_time = time.time()
        # sim_instances = [create_sim_custom_dummy(self.config, dummy_sim_path, id, inet_path) for id in range(self.nr_of_sims)] # Dummy omnet.
        sim_instances = [create_sim_inet_lans_dummy(self.config, dummy_sim_path, uuid.uuid4(), inet_path) for _ in range(self.nr_of_sims)] # Dummy inet lans

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

        # Configure siminstances.
        inet_path = self.conf.tryGet("simulation_model", "simulation_model_paths", "inet_path")
        dummy_sim_path = self.conf.tryGet("simulation_model", "simulation_model_paths", "dummy_path")
        sim_instances = [create_sim_inet_lans_dummy_parallel(self.config, dummy_sim_path, sim_id, inet_path) for sim_id in sim_ids]

        uids = {sim_instance.uid: id for id, sim_instance in enumerate(sim_instances)}

        # Run the configured simulation model.
        self.manager.enqueue_tasks(sim_instances)
        self.manager.evaluate_all()

        return uids


    def create_relevant_cluster_config(self):
        platform = self.conf.tryGet("simulation_model", "simulation_model_configuration", "platform")
        if platform == "DAS":
            self.logger.info("Setting up DAS cluster configuration.")
            return WorkflowConfig.create_slurm_cluster_config(
                self.conf.tryGet("simulation_model", "simulation_model_configuration", "jobs"),
                self.conf.tryGet("simulation_model", "simulation_model_configuration", "job_cores"),
                self.conf.tryGet("simulation_model", "simulation_model_configuration", "job_processes"),
                self.conf.tryGet("simulation_model", "simulation_model_configuration", "job_memory")
            )
        elif platform == "local":
            self.logger.info("Setting up local cluster configuration.")
            return WorkflowConfig.create_local_cluster_config(
                self.conf.tryGet("simulation_model", "simulation_model_configuration", "num_workers"),
                self.conf.tryGet("simulation_model", "simulation_model_configuration", "threads_per_worker")
            )
        else:
            raise ValueError(f"Unknown platform value {platform} provided in coordinator config.")


    def remove_simulation_run_dirs(self):
        self.logger.info("Removing simulation run templates in dummy path.")
        sim_dummy_directory = self.conf.tryGet("simulation_model", "simulation_model_paths", "dummy_path")
        pattern = "custom_dummy_*"
        simulation_run_dir_pattern = os.path.join(sim_dummy_directory, pattern)
        matching_sim_dummy_dir = [path for path in glob.glob(simulation_run_dir_pattern) if os.path.isdir(path)]
        for directory in matching_sim_dummy_dir:
            shutil.rmtree(directory)


    # TODO: Move to data module.
    def store_design_point_metrics(self, latency_df, cost_df, sim_uid):

        # TODO: Make file name sim_instance dependent such that correct storage location is configured and data is stored with other results.
        # Define the file output path.
        dir_design_points_metrics_output = self.conf.tryGet("output_paths", "design_points_metrics_output")
        os.makedirs(dir_design_points_metrics_output, exist_ok=True)
        append_design_points_metric_output_file = os.path.join(dir_design_points_metrics_output, "design_point_metrics.csv")

        # Determine weighted metric values.
        adjusted_latency = latency_df['mean'].mean() * self.conf.tryGet("fitness_config", "weight_latency")
        adjusted_network_cost = cost_df['value'].astype(float).sum() * self.conf.tryGet("fitness_config", "weight_cost")

        # Append the adjusted values to the design points metrics storage.
        append_df = pd.DataFrame([[sim_uid, adjusted_latency, adjusted_network_cost]],
                                columns=['SimulationID', 'AdjustedLatency', 'AdjustedNetworkCost'])
        append_df.to_csv(append_design_points_metric_output_file, mode='a', header=not os.path.exists(append_design_points_metric_output_file), index=False)


    '''
        Perform the parameter tuning workflow.
    '''
    def parameter_tuning_workflow(self):
        self.logger.info("Performing the parameter tuning workflow.")
        parameter_names = self.conf.tryGet("parameter_tuning", "parameter_names")
        parameter_ranges = self.conf.tryGet("parameter_tuning", "parameter_ranges")

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
                    {parameter_name: parameter_value}
                )

        # Run the simulation model.
        uids = self.run_multiple_simulation_configuration(sim_ids)

        for sim_uid in uids.keys():
            self.determine_sim_instance_parameter_tuning_results(sim_uid)

        sim_ids.clear()
        self.remove_simulation_run_dirs()


    '''
        Determine the parameter tuning results for the simulation run.

        Args:
            sim_uid: The unique identifier of the simulation run.
    '''
    def determine_sim_instance_parameter_tuning_results(self, sim_uid):
        self.logger.info(f"Determining the parameter tuning results for siminstance {sim_uid}.")

        # TODO: Change scavetool output filename to something more descriptive.
        csv_file_path = os.path.join(self.data_path, "results", sim_uid, "x.csv")
        df = pd.read_csv(csv_file_path)

        # TODO: Make the following code more generic and less hardcoded.
        cli_df = df[df["module"].fillna("").str.endswith(".cli")]
        self.logger.debug(f"CLI dataframe head:\n{cli_df.head(40)}")
        latency_df = cli_df[(cli_df["type"] == "statistic") & (cli_df["name"].str.startswith("endToEndDelay"))]
        packet_df = cli_df[(cli_df['type'] == "scalar") & (cli_df['name'].str.startswith("packet"))]

        # Remove the csv file.
        if self.remove_sim_instance_output:
            self.logger.info(f"Removing the output file for simistance {sim_uid}")
            os.remove(csv_file_path)

        self.save_parameter_tuning_results(latency_df, packet_df, sim_uid)


    '''
        Save the parameter tuning results from the simulation run.

        Args:
            latency_df: The dataframe containing the latency statistics of the simulation run.
            packet_df: The dataframe containing the packet statistics of the simulation run.
            sim_uid: The unique identifier of the simulation run.
    '''
    # TODO: Move to data module.
    def save_parameter_tuning_results(self, latency_df, packet_df, sim_uid):

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
        append_df = pd.DataFrame([[sim_uid, count_packets_sent, packets_sent_sum, count_packets_received, packets_received_sum, mean_end_to_end_delay, stddev_end_to_end_delay, min_end_to_end_delay, max_end_to_end_delay]],
                                columns=['SimulationID', 'CountPacketsSent', 'PacketsSentSum', 'CountPacketsReceived', 'PacketsReceivedSum', 'MeanEndToEndDelay', 'StddevEndToEndDelay', 'MinEndToEndDelay', 'MaxEndToEndDelay'])
        append_df.to_csv(self.parameter_tuning_file, mode='a', header=not os.path.exists(self.parameter_tuning_file), index=False)


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
    def write_new_ini_file_new(old_file_path, new_file_path, configurations, nr_of_backbone_switches):
        with open(old_file_path, 'r') as old_file, open(new_file_path, 'w') as new_file:
            for line in old_file:
                # Update the number of backbone switches accordingly
                if line.startswith("LargeNet.n"):
                    new_file.write(f"LargeNet.n = {nr_of_backbone_switches}   # number of switches on backbone\n")
                else:
                    new_file.write(line)

            new_file.write("\n# Custom parameters\n")

            for configuration in configurations:
                config_pattern = configuration["config_pattern"]
                value = configuration["value"]
                new_file.write(f"{config_pattern} = {value}\n")

    @staticmethod
    def write_sim_instance_ini_file(template_ini_file_path, sim_instance_ini_file_path, configurations):
        configuration_patterns = configurations.keys()
        with open(template_ini_file_path, 'r') as template_ini_file, open(sim_instance_ini_file_path, 'w') as sim_instance_ini_file:
            for line in template_ini_file:
                line_written = False
                for configuration_pattern in configuration_patterns:
                    if line.startswith(configuration_pattern):
                        configuration_value = configurations[configuration_pattern]
                        sim_instance_ini_file.write(f"{configuration_pattern} = {configuration_value}\n")
                        line_written = True
                        break
                if not line_written:
                    sim_instance_ini_file.write(line)

    @staticmethod
    def write_new_ini_file(old_file_path, new_file_path, param_dict):
        with open(old_file_path, 'r') as old_file, open(new_file_path, 'w') as new_file:
            for line in old_file:
                updated_line = line
                for param, value in param_dict.items():
                    if line.startswith(param):
                        updated_line = f"{param} = {value}\n"
                new_file.write(updated_line)

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


if __name__ == "__main__":

    coordinator = HeuristicSimulationCoordinator("/home/lvdwater/hyper-heuristic-dse-2.0/config/coordinator.json")
