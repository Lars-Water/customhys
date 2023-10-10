import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
    Save the best fitness values at every iteration step as
    a plot for fitness over iteration step.

    hist_values: Historicl values of run metaheuristic
    store_path:  Path to store the plot.
'''


def save_simulation_fitness(hist_values, store_path):

    # Create directory if it doesn't exist
    if not os.path.exists(os.path.dirname(store_path)):
        os.makedirs(os.path.dirname(store_path))
    
    # # From local machine, but I do not really know why anymore. CHECK Notebook if I have
    # # local machine again!
    # # Invert the historical values as the fitness values are the inverse.
    # inverted_values = [1 / x for x in hist_values['fitness']]

    # Extract the 'fitness' values into a list and convert the arrays to regular numbers
    fitness_values = [x for x in hist_values['fitness']]

    # Create the plot
    plt.figure()
    plt.plot(fitness_values, lw=2)
    plt.xlabel('Iteration')
    plt.ylabel('Simulation exection time')
    plt.xlim(0, len(fitness_values) - 1)
    plt.ylim(0, max(fitness_values) + 1)
    plt.xticks(np.arange(0, len(fitness_values), step=1))
    plt.grid()
    plt.savefig("/workspaces/hyper-heuristic-dse-2.0/data/processed/my_figure.png")


'''
    Read the csv file in path /workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/workflow/results/ab5f71b5e22b1928ce6aeae0ae2aaadb/tictoc.csv to a dataframe. 
    
    Define the dataframe such that it contains all rows with column 'type' having value 'statistic'. Skip on columns 'underflows,overflows,binedges,binvalues'.
'''
def save_hop_count():
    df = pd.read_csv("/workspaces/hyper-heuristic-dse-2.0/src/external/simulation_model/workflow/results/ab5f71b5e22b1928ce6aeae0ae2aaadb/tictoc.csv")
    df = df[df.type == 'statistic'].drop(columns=['underflows','overflows','binedges','binvalues'])
    
    # For every row, make a box plot with the values 'mean, stddev, min, max'. The plot title should be the value of column 'module'
    for index, row in df.iterrows():
        plt.figure()
        plt.boxplot([row['mean'], row['stddev'], row['min'], row['max']])
        plt.title(f'Hop count distribution - {row["module"]}')
        plt.savefig("/workspaces/hyper-heuristic-dse-2.0/data/processed/" + row['module'] + ".png")


if __name__ == "__main__":
    save_hop_count()
    # save_simulation_fitness()
