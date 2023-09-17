import sys
sys.path.append('/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model')
import experiments

# from import experiment_seq
# from src.manager import Manager
# from src.utils.config_creator import WorkflowConfig

# from tools import tools

import subprocess

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
    
    def evaluate_simulation_instances(self, config):
        try:
            # Configure the simulation
            pass
            
            # Run the simulation and capture the output
            self.simulation_manager.enqueue_tasks(sim_instances)
            evaluated_sim_instances = self.simulation_manager.evaluate_all()
            
        except Exception as e:
            print(f"An error occurred while running the simulation: {e}")
            return None
    
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

    # # Temporarily replace command-line arguments
    # sys.argv = [
    #     'experiment_seq.py', 
    #     '--data_path=/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/experiments/', 
    #     '--inet_path=/workspaces/omnetpp-6.0.1/inet4.5/', 
    #     '--sims_path=/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/sims/',
    #     '--dummy_path=/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/sims/',
    #     '--platform=local'
    # ]
    # Temporarily replace command-line arguments
    sys.argv = [
        'experiment_seq.py', 
        '--experiments_path=/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/experiments/', 
        '--inet_path=/workspaces/omnetpp-6.0.1/inet4.5/', 
        '--sims_path=/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/sims/',
        '--dummy_path=/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/sims/',
        '--platform=local'
    ]

    # Run the main function of the target script
    experiments.main()

    # Restore original command-line arguments
    sys.argv = original_argv    

    # # Initate simulation-heuristic coordinator.
    # coordinator = HeuristicSimulationCoordinator()
    
    # # Configure model boundaries
    # boundaries = {"param1": "value1", "param2": "value2"}
    # coordinator.configure_model_boundaries(boundaries)
    
    # # Run simulation
    # config = {"setting1": "value1", "setting2": "value2"}
    # simulation_output = coordinator.call_simulation_run(config)
    
    # # Process output
    # coordinator.process_simulation_output(simulation_output)
    
    # coordinator.simulation_manager.shutdown()
