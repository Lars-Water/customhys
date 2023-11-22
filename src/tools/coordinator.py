from tools.logger import logger
from pathlib import Path
from time import sleep

import argparse
import os
import shutil
import sys
import pandas as pd
import numpy as np
import subprocess
import time

sys.path.append('/home/larry/hyper-heuristic-dse-2.0/src/external/simulation_model')
import experiments
from experiments import create_sim_custom_dummy, create_sim_inet_lans_dummy
import experiment_campaign
from src.manager import Manager
from src.utils.config_reader import Config
from src.utils.config_creator import WorkflowConfig


class HeuristicSimulationCoordinator:
    
    def __init__(self, config_path):

        # Set up the logger.
        os.makedirs("/home/larry/hyper-heuristic-dse-2.0/data/logs/coordinator", exist_ok=True)
        self.logger = logger("coordinator", Path("/home/larry/hyper-heuristic-dse-2.0/data/logs/coordinator"))

        # TODO: Change the following hardcoded parameter definition to reading a config file.
        # Define params to set up the Manager.
        experiments_path = "/home/larry/hyper-heuristic-dse-2.0/src/external/simulation_model/experiments"
        model = "custom"
        num_nodes = 1
        num_workers = 1
        num_sims = 1
        time_stamp = time.strftime("%Y%m%d_%H%M%S")

        self.data_path = os.path.join(experiments_path, "data", "campaign_{}".format(model), "n{}_w{}_s{}".format(str(num_nodes), str(num_workers), str(num_sims)), time_stamp)

        sims_path = "/home/larry/hyper-heuristic-dse-2.0/src/external/simulation_model/sims"

        design_queues = [WorkflowConfig.create_design_point_queue_config("base", 0, "FIFO")]

        num_threads_per_worker = int(6 / num_workers)

        cluster_config = WorkflowConfig.create_local_cluster_config(num_workers, num_threads_per_worker)

        # Define params for configuration file creation.
        workflow_config_file = os.path.join(self.data_path, "config.json")
        workflow_results_folder = os.path.join(self.data_path, "results")
        workflow_logs_folder = os.path.join(self.data_path, "logs")
        workflow_runtime_folder = os.path.join(self.data_path, "runtime")

        # Set up workflow configuration file.
        self.logger.info("Setting up workflow configuration file...")
        self.workflow_config = WorkflowConfig(sims_path, "config", "run_sim", "results", "logs", "out",
                            workflow_results_folder, workflow_logs_folder, workflow_runtime_folder, design_queues, "md5-files", cluster_config)
        self.config = self.workflow_config.conf()
        self.workflow_config.write_conf(workflow_config_file)

        # Set up the Manager.
        self.logger.info("Setting up the Manager...")
        self.manager = Manager(workflow_config_file, workflow_logs_folder)

        # Read the config file.
        self.logger.info("Reading coordinator config file...")
        coordinator_path = "/home/larry/hyper-heuristic-dse-2.0/coordinator"
        coordinator_config_file = "/home/larry/hyper-heuristic-dse-2.0/config/config_coordinator.json"
        self.conf = Config(Path(coordinator_config_file), Path(coordinator_path), "coordinator_config_manager")

        # Predefine the parameters for the execution times.
        self.coordinator_execution_time = 0
        self.simulation_execution_time = 0


        self.uids = []


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
        
        self.logger.debug(f"Simulation run with config values: {config_values}")
        
        # Set the values of the parameters in the simulation model.
        configurations = self.set_param_values(config_values)
        
        src_dir = self.conf.tryGet("dummy_sim_src_dir")
        dest_dir = self.conf.tryGet("dummy_sim_dest_dir")
        
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

        # Transform the outputted scalar files into csv format.
        self.transform_scalar_files(uid)

        # Collect the simulation stats from the simulation run.
        simulation_metrics = self.obtain_simulation_stats(uid)

        fitness_config = self.conf.tryGet("fitness_config")
        fitness_value = fitfunc(fitness_config, simulation_metrics)        

        # End timer for simulation run.
        end_time = time.time()

        # Add simulation run time to total coordinator time.
        self.coordinator_execution_time += end_time - start_time
        
        return fitness_value

    
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
        params = self.conf.tryGet("simulation_model", "params")
        
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
        Transform the scalar files generated by the simulation model into a csv format.
    '''
    def transform_scalar_files(self, uid):
        
        # Add the omnetpp bin directory to PATH environment variables.
        self.logger.info("Adding omnetpp bin directory to PATH environment variables")
        new_path = os.path.join(self.conf.tryGet("omnetpp_base"), "bin")
        current_path = os.environ.get('PATH', '')
        if new_path not in current_path:
            os.environ['PATH'] = new_path + os.pathsep + current_path
        
        # # Automated workflow.
        sim_results_directory = os.path.join(self.data_path, "results", uid)
        # # Manual operation from running local workflow.
        # sim_results_directory = os.path.join("/home/larry/hyper-heuristic-dse-2.0/src/external/simulation_model/workflow", "results", uid)
        
        scavetool_command = "opp_scavetool export -F CSV-R -o x.csv *.sca"

        # Attempt to transform the scalar files in the given results directory.
        self.logger.info(f"Transforming scalar files to csv in directory: {sim_results_directory}")
        try:
            subprocess.run(scavetool_command, shell=True, cwd=sim_results_directory, check=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error executing the command: {e}")

        # Remove the scalar files in the given results directory.
        self.logger.info(f"Removing scalar files in directory: {sim_results_directory}")
        for file in os.listdir(sim_results_directory):
            if file.endswith(".sca"):
                os.remove(os.path.join(sim_results_directory, file))
    

    '''
        Collect the simulation stats from the simulation run.

        Args:
            uid: The unique identifier of the simulation run.
    '''
    def obtain_simulation_stats(self, uid):

        # Load the transformed outputted data from the simulation run.

        # # Automated workflow.
        csv_file_path = os.path.join(self.data_path, "results", uid, "x.csv")
        # # Manual operation from running local workflow.
        # csv_file_path = os.path.join("/home/larry/hyper-heuristic-dse-2.0/src/external/simulation_model/workflow", "results", uid, "x.csv")

        df = pd.read_csv(csv_file_path)

        # # Filter rows with 'type' column equal to 'statistic' and drop unnecessary columns
        # df = df[df['type'] == 'statistic'].drop(columns=['underflows', 'overflows', 'binedges', 'binvalues'])
        
        # Get the end-to-end delay statistics that are gathered by default in INET LANS simulation runs.
        latency_df = df[df["type"] == "histogram"]
        latency_df = latency_df[latency_df["module"].str.endswith(".cli")]
        latency_df = latency_df[latency_df["name"].str.startswith("endToEndDelay")]

        # Get the cost of all components with a cost paramater in the simulation model.
        cost_df = df[df['type'] == "param"]
        cost_df = cost_df[cost_df["name"] == "cost"]

        # Remove the csv file.
        os.remove(csv_file_path)

        # return the average of the column 'mean' in the dataframe.
        return {
            "latency": latency_df['mean'].mean(), 
            "network_cost": cost_df['value'].astype(float).sum()
        }
    

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
        num_sims = self.conf.tryGet("num_sims")
        inet_path = self.conf.tryGet("inet_path")
        dummy_sim_path = self.conf.tryGet("dummy_path")

        start_time = time.time()
        # sim_instances = [create_sim_custom_dummy(self.config, dummy_sim_path, id, inet_path) for id in range(num_sims)] # Dummy omnet.
        sim_instances = [create_sim_inet_lans_dummy(self.config, dummy_sim_path, id, inet_path) for id in range(num_sims)] # Dummy inet lans

        uid = sim_instances[0].uid
        if uid in self.uids:
            print(f"UID {uid} already exists in {self.uids}")
        else:            
            self.uids.append(uid)

        # Run the configured simulation model.
        self.manager.enqueue_tasks(sim_instances)

        evaluated_sim_instances = self.manager.evaluate_all()
        uid = evaluated_sim_instances[0].uid
        
        # # Keep trying to retrieve the unique identifier of the simulation run until it succeeds.
        # uid = None
        # while uid is None:
        #     try:
        #         evaluated_sim_instances = self.manager.evaluate_all()
        #         uid = evaluated_sim_instances[0].uid
        #     except:
        #         print("Could not find sim_instance uid.")
        #         time.sleep(1)

        end_time = time.time()

        # TODO: Change the determined execution time to obtaining it from the simulation model runtime folder.
        # Aggregate simulation time to a variable of total simulation time.
        self.simulation_execution_time += end_time - start_time

        return uid

    
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

    @staticmethod
    def run_experiment_campaign(model, num_nodes, num_workers, num_sims, time_stamp, experiments_path):

        # Backup the original command-line arguments
        original_argv = sys.argv
        original_directory = os.getcwd()

        # define flag values for the experiment campaign.
        data_path = os.path.join(experiments_path, "data", "campaign_{}".format(model), "n{}_w{}_s{}".format(num_nodes, num_workers, num_sims), time_stamp)
        
        # Temporarily replace command-line arguments for experiment_campaign.py
        sys.argv = [
            'experiment_campaign.py', 
            '--data_path=' + data_path,
            '--inet_path=/home/larry/omnetpp-6.0.1/inet4.5/', 
            '--sims_path=/home/larry/hyper-heuristic-dse-2.0/src/external/simulation_model/sims/',
            '--dummy_path=/home/larry/hyper-heuristic-dse-2.0/src/external/simulation_model/sims/',
            '--platform=local',
            '--model=' + model,
            '--num_nodes=' + num_nodes,
            '--num_workers=' + num_workers,
            '--num_sims=' + num_sims
        ]

        # Temporarily change the working directory
        os.chdir('/home/larry/hyper-heuristic-dse-2.0/src/external/simulation_model/')

        # Run the main function of the target script
        uid = experiment_campaign.main()

        # Restore original command-line arguments and working directory
        sys.argv = original_argv
        os.chdir(original_directory)

        return uid

    @staticmethod
    def campaign_run_collect(model, num_nodes, num_workers, num_sims, time_stamp, experiments_path):

        # Backup the original command-line arguments
        original_argv = sys.argv
        original_directory = os.getcwd()

        # Temporarily replace command-line arguments for experiments.campaign_run_collect()
        sys.argv = [
            'experiments.py', 
            f'--experiments_path={experiments_path}'
        ]

        # Set up the argument parser for experiments.campaign_run_collect()
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument("--experiments_path", default="experiments/", help="specifies the base path of the project")
        args = arg_parser.parse_args()
        results_path = os.path.join(args.experiments_path, "results", "campaign_{}".format(model), "n{}_w{}_s{}".format(num_nodes, num_workers, num_sims), time_stamp)

        # Temporarily change the working directory
        os.chdir('/home/larry/hyper-heuristic-dse-2.0/src/external/simulation_model/')

        # Run the data collection function of the target script.
        experiments.campaign_run_collect(args, model, num_nodes, num_workers, num_sims, time_stamp)

        # Restore original command-line arguments and working directory
        sys.argv = original_argv
        os.chdir(original_directory)

        return os.path.join(results_path, "sims_runtime.csv")


if __name__ == "__main__":

    coordinator = HeuristicSimulationCoordinator("/home/larry/hyper-heuristic-dse-2.0/config/coordinator.json")
