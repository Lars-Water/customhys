import os
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
        return simulation_model.create_sim_pdes_comm(self.config["dp"]["workflowConfigFile"], self.config["dp"]["dummy_sim_path"], self.config["dp"]["inet_base"], id=0)
    
    def setup_workflow_config(self):
        pass
    
    def setup_simulation_manager(self):
        self.simulation_manager = Manager(self.config["manager"]["workflowConfigFile"], self.config["manager"]["workflowLogsFolder"])
    
    def configure_model_boundaries(self, boundaries):
        # Configure the CUSTOMHys framework
        self.customhys.configure_boundaries(boundaries)
    
    def process_simulation_output(self, output):
        # Process the simulation output and use it as feedback for the hyper-heuristic
        feedback = self.omnet.process_output(output)
        self.customhys.use_feedback(feedback)

# Example usage
if __name__ == "__main__":

    # Backup the original command-line arguments
    original_argv = sys.argv
    original_directory = os.getcwd()
    
    # # Temporarily replace command-line arguments
    # sys.argv = [
    #     'experiments.py', 
    #     '--experiments_path=/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/experiments/',
    #     '--inet_path=/workspaces/omnetpp-6.0.1/inet4.5/', 
    #     '--sims_path=/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/sims/',
    #     '--dummy_path=/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/sims/',
    #     '--platform=local'
    # ]

    # Temporarily replace command-line arguments
    sys.argv = [
        'experiment_campaign.py', 
        '--data_path=', 
        '--inet_path=/workspaces/omnetpp-6.0.1/inet4.5/', 
        '--sims_path=/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/sims/',
        '--dummy_path=/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/sims/',
        '--platform=local',
        '--model=',
        '--num_nodes=',
        '--num_workers=',
        '--num_sims='
    ]

    # Temporarily change the working directory
    os.chdir('/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/')

    # Run the main function of the target script
    # experiments.main()
    experiment_campaign.main()

    # Restore original command-line arguments and working directory
    sys.argv = original_argv
    os.chdir(original_directory)
