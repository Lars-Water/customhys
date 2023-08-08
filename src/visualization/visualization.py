import matplotlib.pyplot as plt

'''
    Save the best fitness values at every iteration step as
    a plot for fitness over iteration step.

    hist_values: Historicl values of run metaheuristic
    store_path:  Path to store the plot.
'''


def save_simulation_fitness(hist_values, store_path):

    # Invert the historical values as the fitness values are the inverse.
    inverted_values = [1 / x for x in hist_values['fitness']]

    plt.figure()
    plt.plot(inverted_values, lw=2)
    plt.xlabel('Iteration'), plt.ylabel('Fitness')
    plt.tight_layout()
    plt.savefig(store_path)
