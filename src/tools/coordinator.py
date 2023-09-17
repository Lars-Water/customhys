import os
import shutil
import sys

sys.path.append('/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model')
import experiments
import experiment_campaign


class HeuristicSimulationCoordinator:
    
    def __init__(self, config_path):
        # test_path = "/workspaces/hyper-heuristic-dse-2.0/config/coordinator.json"
        # self.config = tools.load_config(test_path)
        # self.config = tools.load_config(config_path)
        pass
    
    def generate_design_point(self):
        #return simulation_model.create_sim_pdes_comm(self.config["dp"]["workflowConfigFile"], self.config["dp"]["dummy_sim_path"], self.config["dp"]["inet_base"], id=0)
        pass
    
    def setup_workflow_config(self):
        pass
    
    def setup_simulation_manager(self):
        #self.simulation_manager = Manager(self.config["manager"]["workflowConfigFile"], self.config["manager"]["workflowLogsFolder"])
        pass
    
    def configure_model_boundaries(self, boundaries):
        # Configure the CUSTOMHys framework
        # self.customhys.configure_boundaries(boundaries)
        pass
    
    def process_simulation_output(self, output):
        # Process the simulation output and use it as feedback for the hyper-heuristic
        # feedback = self.omnet.process_output(output)
        # self.customhys.use_feedback(feedback)
        pass

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


if __name__ == "__main__":

    coordinator = HeuristicSimulationCoordinator("/workspaces/hyper-heuristic-dse-2.0/config/coordinator.json")

    # Dynamic parameters as a dictionary
    param_dict = {
        "*.sDelay": "exponential(0.2s)",
        "*.tandemQueue[*].queue[*].numInitialJobs": "2000",
        "*.tandemQueue[*].queue[*].serviceTime": "exponential(0.2s)",
        "*.tandemQueue[*].qDelay": "exponential(0.2s)",
        "*.tandemQueue[*].switch[*].retain": "0.2"
    }
    
    src_dir = '/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/sims/dummy_sim_pdes_communicate'
    dest_dir = '/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/sims/custom_dummy'

    # Update existing file.
    # ini_file_path = os.path.join(src_dir, 'communicate_intensive.ini')
    # coordinator.update_file_params(ini_file_path, param_dict)
    
    # Delete destination directory if it exists
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    # Duplicate a new custom dummy sim directory and ignore the given filename.
    shutil.copytree(src_dir, dest_dir, ignore=coordinator.ignore_file('communicate_intensive.ini'))

    old_ini_file_path = os.path.join(src_dir, 'communicate_intensive.ini')
    new_ini_file_path = os.path.join(dest_dir, 'communicate_intensive.ini')

    coordinator.write_new_ini_file(old_ini_file_path, new_ini_file_path, param_dict)


    # # Backup the original command-line arguments
    # original_argv = sys.argv
    # original_directory = os.getcwd()
    
    # # # Temporarily replace command-line arguments
    # # sys.argv = [
    # #     'experiments.py', 
    # #     '--experiments_path=/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/experiments/',
    # #     '--inet_path=/workspaces/omnetpp-6.0.1/inet4.5/', 
    # #     '--sims_path=/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/sims/',
    # #     '--dummy_path=/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/sims/',
    # #     '--platform=local'
    # # ]

    # # Temporarily replace command-line arguments
    # sys.argv = [
    #     'experiment_campaign.py', 
    #     '--data_path=', 
    #     '--inet_path=/workspaces/omnetpp-6.0.1/inet4.5/', 
    #     '--sims_path=/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/sims/',
    #     '--dummy_path=/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/sims/',
    #     '--platform=local',
    #     '--model=',
    #     '--num_nodes=',
    #     '--num_workers=',
    #     '--num_sims='
    # ]

    # # Temporarily change the working directory
    # os.chdir('/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/')

    # # Run the main function of the target script
    # # experiments.main()
    # experiment_campaign.main()

    # # Restore original command-line arguments and working directory
    # sys.argv = original_argv
    # os.chdir(original_directory)
