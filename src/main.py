import tools.tools as tools
import models.model as model
import visualization.visualization as visualization

from external.SimulationModel.SimulationModel import SimulationModel

from external.CUSTOMHys.customhys import metaheuristic as mh


def main(base_path):

    # Initialize simulation model
    simulation_model = SimulationModel()

    # Create fake CQN formulation from config file and simulation model.
    config = tools.load_config(base_path + '/config/config.json')
    cqn_instance = model.generate_instance(4,
                                           config['Simulation']['cqn'],
                                           simulation_model.run)
    prob = cqn_instance.get_formatted_problem()

    # Create metaheuristic search operator for uniform random search.
    # For all default operators see "~data/external/default_operators.txt"
    heur = [(
        'random_search',
        {
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'greedy'
    )]

    # Generate a metaheuristic search method for the CQN model.
    met = mh.Metaheuristic(prob, heur, num_agents=30, num_iterations=100)
    met.verbose = True

    # Run the metaheuristic on the problem. The fitness value is calculated at
    # every iteration step by the run function in the SimulationModel class.
    met.run()

    # Save the best fitness value for every iteration in a plot.
    visualization.save_simulation_fitness(met.historical,
                                          base_path + '/data/processed/Fake_CQN_fitness_over_iteration.png')

    return


if __name__ == "__main__":
    base_path = '/mnt/c/users/larry/Desktop/SE_Thesis/hh-dse-2.0/Search_Method'
    main(base_path)
