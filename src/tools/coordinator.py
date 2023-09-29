import argparse
import os
import shutil
import sys
import time
import pandas as pd

sys.path.append('/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model')
import experiments
import experiment_campaign


class HeuristicSimulationCoordinator:
    
    def __init__(self, config_path):
        pass
    
    def simulation_run(self, config_values):
        
        # Set the values of the parameters in the simulation model.
        param_dict = self.set_param_values(config_values)
        
        # Duplicate the preferred dummy_sim directory to the custom directory.
        src_dir = '/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/sims/dummy_sim_pdes_communicate'
        dest_dir = '/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/sims/custom_dummy'
        self.duplicate_directory(src_dir, dest_dir, "communicate_intensive.ini")

        # Write an updated version of the ignored param value file from the directory that was just duplicated.
        old_ini_file_path = os.path.join(src_dir, "communicate_intensive.ini")
        new_ini_file_path = os.path.join(dest_dir, "communicate_intensive.ini")
        self.write_new_ini_file(
            old_ini_file_path, 
            new_ini_file_path, 
            param_dict
        )

        # Define flag values for the experiment campaign.
        # TODO: Make these configurable.
        model = "tictoc"
        num_nodes = str(1)
        num_workers = str(1)
        num_sims = str(1)
        time_stamp = time.strftime("%Y%m%d_%H%M%S")
        experiments_path = "/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/experiments"
        
        # Run the experiment campaign.
        self.run_experiment_campaign(model, num_nodes, num_workers, num_sims, time_stamp, experiments_path)

        # Collect the sim_runtime.
        sim_runtime_csv_file_path = self.campaign_run_collect(model, num_nodes, num_workers, num_sims, time_stamp, experiments_path)
        sim_exec_time = self.process_simulation_output(sim_runtime_csv_file_path, 'simulation.sim_exec_time')

        return sim_exec_time
    
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

    '''
    Set the values of the parameters in the simulation model.

    Args:
        config_values: A list of values for the parameters in the simulation model.
    '''
    @staticmethod
    def set_param_values(config_values):

        # Boundary configuration.
        boundaries = {
            "*.sDelay": [0.1, 0.5],
            "*.tandemQueue[*].queue[*].serviceTime": [0.1, 0.5],
            "*.tandemQueue[*].qDelay": [0.1, 0.5],
            "*.tandemQueue[*].switch[*].retain": [0.1, 0.5]
        }

        # Dynamic parameters mapped to value distriution.
        param_dict = {
            "*.sDelay": "exponential",
            "*.tandemQueue[*].queue[*].serviceTime": "exponential",
            "*.tandemQueue[*].qDelay": "exponential",
            "*.tandemQueue[*].switch[*].retain": "uniform"
        }

        # Update param_dict based on provided config values following the order of the boundaries.
        for idx, (param_key, _) in enumerate(boundaries.items()):
            distribution = param_dict.get(param_key, None)
            config_value = config_values[idx]  # Fetch the corresponding value from the confiq_values
            
            if distribution == "exponential":
                param_dict[param_key] = f"exponential({config_value}s)"
            elif distribution == "uniform":
                param_dict[param_key] = str(config_value)
            # Add other distributions here...
        
        return param_dict
    
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
            '--inet_path=/workspaces/omnetpp-6.0.1/inet4.5/', 
            '--sims_path=/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/sims/',
            '--dummy_path=/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/sims/',
            '--platform=local',
            '--model=' + model,
            '--num_nodes=' + num_nodes,
            '--num_workers=' + num_workers,
            '--num_sims=' + num_sims
        ]

        # Temporarily change the working directory
        os.chdir('/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/')

        # Run the main function of the target script
        experiment_campaign.main()

        # Restore original command-line arguments and working directory
        sys.argv = original_argv
        os.chdir(original_directory)

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
        os.chdir('/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/')

        # Run the data collection function of the target script.
        experiments.campaign_run_collect(args, model, num_nodes, num_workers, num_sims, time_stamp)

        # Restore original command-line arguments and working directory
        sys.argv = original_argv
        os.chdir(original_directory)

        return os.path.join(results_path, "sims_runtime.csv")


if __name__ == "__main__":

    coordinator = HeuristicSimulationCoordinator("/workspaces/hyper-heuristic-dse-2.0/config/coordinator.json")

    # # Backup the original command-line arguments
    # original_argv = sys.argv
    # original_directory = os.getcwd()
    
    # # Temporarily replace command-line arguments for experiments.py
    # sys.argv = [
    #     'experiments.py', 
    #     '--experiments_path=/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/experiments/',
    #     '--inet_path=/workspaces/omnetpp-6.0.1/inet4.5/', 
    #     '--sims_path=/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/sims/',
    #     '--dummy_path=/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/sims/',
    #     '--platform=local'
    # ]

    # # Temporarily change the working directory
    # os.chdir('/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/')

    # # Run the main function of the target script
    # experiments.main()

    # # Restore original command-line arguments and working directory
    # sys.argv = original_argv
    # os.chdir(original_directory)
