import os
import numpy as np

import tools.tools as tools
import tools.coordinator as coordinator
import models.model as model
import visualization.visualization as visualization

# from external.CUSTOMHys.customhys import metaheuristic as mh
# from external.CUSTOMHys.customhys import hyperheuristic as hh

from customhys import metaheuristic as mh
# from customhys import hyperheuristic as hh

def run_mh(heur_sim_coordinator, heuristics_collection, num_agents, num_iterations, verbose=False):

    for run in [6]:
        subnet_structure = dict()
        subnet_structure["boundaries"] = {
                f"cable_{i}": [0, 1] for i in range(1, run + 1)
            }
        subnet_structure["optimal_solution"] = [0.9] * run,
        subnet_structure["optimal_fitness"] = 30

        cqn_instance = model.generate_instance(
            run,
            subnet_structure,
            heur_sim_coordinator.simulation_run
        )
        prob = cqn_instance.get_formatted_problem()

        # Generate a metaheuristic search method for the CQN model.
        met = mh.Metaheuristic(prob, heuristics_collection, num_agents=num_agents, num_iterations=num_iterations)
        met.verbose = verbose

        # Run the metaheuristic on the problem. The fitness value is calculated at
        # every iteration step by the run function in the SimulationModel class.
        met.run()

        # # Save the best fitness value for every iteration in a plot.
        # data_path = "data/processed/"
        # visualization.save_simulation_fitness(
        #     met.historical, 
        #     os.path.join(base_path, data_path, "test_run_heuristic_simulation_workflow.png"), 
        #     "Latency and cost evaluation of INET-LANS model",
        #     f"inet_lans_lat_cost_{str(run)}"
        # )

    return

def run_hh(heur_sim_coordinator, heuristics_collection):
    
    # Setup parameters for the hyperheuristic
    hh_parameters = dict(
    cardinality=3,  # Max. numb. of SOs in MHs, lvl:1
    cardinality_min=1,  # Min. numb. of SOs in MHs, lvl:1
    num_iterations=100,  # Iterations an MH performs, lvl:1
    num_agents=30,  # Agents in population,     lvl:1
    as_mh=True,  # HH sequence as a MH?,     lvl:2
    num_replicas=30,  # Replicas per each MH,     lvl:2
    num_steps=50,  # Trials per HH step,       lvl:2
    stagnation_percentage=0.37,  # Stagnation percentage,    lvl:2
    max_temperature=100,  # Initial temperature (SA), lvl:2
    min_temperature=1e-6,  # Min temperature (SA),     lvl:2
    cooling_rate=1e-3,  # Cooling rate (SA),        lvl:2
    temperature_scheme='fast',  # Temperature updating (SA),lvl:2
    acceptance_scheme='exponential',  # Acceptance mode,          lvl:2
    allow_weight_matrix=True,  # Weight matrix,            lvl:2
    trial_overflow=False,  # Trial overflow policy,    lvl:2
    learnt_dataset=None,  # If it is a learnt dataset related with the heuristic space
    repeat_operators=True,  # Allow repeating SOs inSeq,lvl:2
    verbose=True,  # Verbose process,          lvl:2
    learning_portion=0.37,  # Percent of seqs to learn  lvl:2
    solver='static')  # Indicate which solver use lvl:1

    for run in [6]:
        subnet_structure = dict()
        subnet_structure["boundaries"] = {
                f"cable_{i}": [0, 1] for i in range(1, run + 1)
            }
        subnet_structure["optimal_solution"] = [0.9] * run,
        subnet_structure["optimal_fitness"] = 30

        cqn_instance = model.generate_instance(
            run,
            subnet_structure,
            heur_sim_coordinator.simulation_run
        )
        prob = cqn_instance.get_formatted_problem()
    
        # Create hyperheuristic object and solve the problem.
        hyp = hh.Hyperheuristic(
            heuristic_space=heuristics_collection,
            problem=prob, 
            parameters=hh_parameters, 
            file_label="INET-LANS"
        )

        best_sol, best_perf, hist_curr, hist_best = hyp.solve()

        print(f"Best solution: {best_sol}")
        print(f"Best performance: {best_perf}")
        print(f"Current history: {hist_curr}")
        print(f"Best history: {hist_best}")

        return

'''
    Collect the simulation stats from the simulation run.

    Args:
        heur_sim_coordinator (HeuristicSimulationCoordinator): The coordinator object for the heuristic simulation workflow.
        uid (str): The unique identifier for the simulation run.
'''
def process_simulated_instance(heur_sim_coordinator, uid):
    heur_sim_coordinator.transform_scalar_files(uid)
    simulation_metrics = heur_sim_coordinator.obtain_simulation_stats(uid)
    fitness_value = heur_sim_coordinator.fitness_evaluation(simulation_metrics)

    return fitness_value

def main(base_path):

    heur_sim_coordinator = coordinator.HeuristicSimulationCoordinator("/home/larry/hyper-heuristic-dse-2.0/config/coordinator.json") # noqa 501
    
    heuristics_collection = [
        (
            'random_search', 
            {
                'scale': 1.0, 
                'distribution': 'uniform'
            }, 
            'greedy'
        )
    ]

    # heuristics_collection = [
    #     ('random_search', {'scale': 1.0, 'distribution': 'uniform'}, 'greedy'),
    #     [('differential_mutation', {'expression': 'rand-to-best-and-current', 'num_rands': 1, 'factor': 1.0}, 'greedy'), ('differential_crossover', {'crossover_rate': 0.2, 'version': 'binomial'}, 'greedy')],
    #     ('gravitational_search', {'gravity': 1.0, 'alpha': 0.02}, 'all')
    # ]

    # Run the metaheuristic search operator.
    run_mh(heur_sim_coordinator, heuristics_collection, 2, 1, verbose=False)

    # # Run the hyperheuristic search operator.
    # run_hh(heur_sim_coordinator, heuristics_collection)

    return heur_sim_coordinator.shutdown_manager()


if __name__ == "__main__":
    # UBUNTU
    base_path = '/home/larry/hyper-heuristic-dse-2.0/'
    # PyCharm
    # base_path = r'C:\Users\larry\Desktop\SE_Thesis\hh-dse-2.0'
    # Codespaces
    # base_path = "/home/larry/hyper-heuristic-dse-2.0/"
    main(base_path)



# # Create CQN formulation from the config file.
    # config = tools.load_config(base_path + '/config/config.json')
    # cqn_instance = model.generate_instance(
    #     3,
    #     config['Simulation']['inet_lans'],
    #     heur_sim_coordinator.simulation_run
    # )
    # prob = cqn_instance.get_formatted_problem()
    
    # # Create INET-LANS formulation from the config file.
    # config = tools.load_config(base_path + '/config/config.json')
    
    # # cqn_instance = model.generate_instance(
    # #     7,
    # #     config['Simulation']['inet_lans'],
    # #     heur_sim_coordinator.simulation_run
    # # )
