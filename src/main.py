import os

import tools.tools as tools
import tools.coordinator as coordinator
import models.model as model
import visualization.visualization as visualization

# from external.CUSTOMHys.customhys import metaheuristic as mh
# from external.CUSTOMHys.customhys import hyperheuristic as hh

from customhys import metaheuristic as mh
# from customhys import hyperheuristic as hh


def main(base_path):

    heur_sim_coordinator = coordinator.HeuristicSimulationCoordinator("/workspaces/hyper-heuristic-dse-2.0/config/coordinator.json") # noqa 501
    
    # # Collect the simulation stats from the simulation run.
    # uid = "491e65ec2f6a2342fe509480d09a5987"
    # # heur_sim_coordinator.transform_scalar_files(uid)
    # simulation_metrics = heur_sim_coordinator.obtain_simulation_stats(uid)
    # fitness_value = heur_sim_coordinator.fitness_evaluation(simulation_metrics)

    # Create CQN formulation from the config file.
    config = tools.load_config(base_path + '/config/config.json')
    cqn_instance = model.generate_instance(
        3,
        config['Simulation']['inet_lans'],
        heur_sim_coordinator.simulation_run
    )
    prob = cqn_instance.get_formatted_problem()

    # Create metaheuristic search operator for uniform random search.
    # For all default operators see "~data/external/default_operators.txt"
    heur = [
        (
            'random_search', 
            {
                'scale': 1.0, 
                'distribution': 'uniform'
            }, 
            'greedy'
        )
    ]

    # # heur = [('differential_mutation', {'expression': 'rand-to-best-and-current', 'num_rands': 1, 'factor': 1.0}, 'greedy'), ('differential_crossover', {'crossover_rate': 0.2, 'version': 'binomial'}, 'greedy')]

    # Generate a metaheuristic search method for the CQN model.
    met = mh.Metaheuristic(prob, heur, num_agents=1, num_iterations=0)
    met.verbose = True

    # Run the metaheuristic on the problem. The fitness value is calculated at
    # every iteration step by the run function in the SimulationModel class.
    met.run()

    # Save the best fitness value for every iteration in a plot.
    data_path = "data/processed/" # noqa 501
    visualization.save_simulation_fitness(met.historical, os.path.join(base_path, data_path, "test_run_heuristic_simulation_workflow.png")) # noqa 501

    # # # Setup parameters for the hyperheuristic
    # # parameters = dict(
    # #     cardinality=3,  # Max. numb. of SOs in MHs, lvl:1
    # #     cardinality_min=1,  # Min. numb. of SOs in MHs, lvl:1
    # #     num_iterations=10,  # Iterations a MH performs, lvl:1
    # #     num_agents=30,  # Agents in population,     lvl:1
    # #     as_mh=True,  # HH sequence as a MH?,     lvl:2
    # #     num_replicas=30,  # Replicas per each MH,     lvl:2
    # #     num_steps=50,  # Trials per HH step,       lvl:2
    # #     stagnation_percentage=0.37,  # Stagnation percentage,    lvl:2
    # #     max_temperature=100,  # Initial temperature (SA), lvl:2
    # #     min_temperature=1e-6,  # Min temperature (SA),     lvl:2
    # #     cooling_rate=1e-3,  # Cooling rate (SA),        lvl:2
    # #     temperature_scheme='fast',  # Temperature updating (SA),lvl:2
    # #     acceptance_scheme='exponential',  # Acceptance mode,          lvl:2
    # #     allow_weight_matrix=True,  # Weight matrix,            lvl:2
    # #     trial_overflow=False,  # Trial overflow policy,    lvl:2
    # #     learnt_dataset=None,  # If it is a learnt dataset related with the heuristic space # noqa 501
    # #     repeat_operators=True,  # Allow repeating SOs inSeq,lvl:2
    # #     verbose=True,  # Verbose process,          lvl:2
    # #     learning_portion=0.37,  # Percent of seqs to learn  lvl:2
    # #     solver='static'
    # # )  # Indicate which solver use lvl:1

    # # # Create hyperheuristic object and solve the problem.
    # # hyp = hh.Hyperheuristic(
    # #     heuristic_space="/mnt/c/users/larry/Desktop/SE_Thesis/hh-dse-2.0/data/external/default_operators.txt", # noqa 501
    # #     problem=prob, parameters=parameters, file_label='Fake-CQN')
    # # best_solution, best_performance, historical_current, historical_best = hyp.solve() # noqa 501

    # # visualization.save_simulation_fitness({"fitness": historical_best},
    # #                                       base_path + '/data/processed/Fake_CQN_fitness_hh.png') # noqa 501

    # heur_sim_coordinator.shutdown_manager()
    
    # return


if __name__ == "__main__":
    # UBUNTU
    # base_path = '/mnt/c/users/larry/Desktop/SE_Thesis/hh-dse-2.0'
    # PyCharm
    # base_path = r'C:\Users\larry\Desktop\SE_Thesis\hh-dse-2.0'
    # Codespaces
    base_path = "/workspaces/hyper-heuristic-dse-2.0/"
    main(base_path)
