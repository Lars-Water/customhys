import os
import numpy as np
import pandas as pd
import glob

from tools.config_reader import Config
from pathlib import Path

import matplotlib.pyplot as plt
plt.rcParams.update({'text.usetex': False})


def bar_execution_times(type="components"):
    base_path = Path(os.getcwd()).parent
    path = os.path.join(base_path, "/data/raw/results/metaheuristic/random_search/")

    # Initialize lists to store execution times for all files
    total_times = []
    heuristic_times = []
    coordinator_times = []
    simulation_times = []
    values_x_axis = []

    # Iterate through all json files
    files = glob.glob(f"{path}*.json")
    files.sort()
    for file in files:
        # Get the path of the file
        path = Path(file)

        # Create a config object with the path of the file
        os.makedirs(os.path.dirname("/home/larry/hyper-heuristic-dse-2.0/logs/vizualization"), exist_ok=True)
        conf = Config(path, Path("/home/larry/hyper-heuristic-dse-2.0/logs/vizualization"), "execution_times")

        # Get the execution times from the config object
        value_x_axis = conf.tryGet("nr_of_components") if type == "components" else conf.tryGet("nr_of_iterations")
        total_execution_time = round(conf.tryGet("metadata", "total_execution_time") / 60, 1)
        heuristic_execution_time = round(conf.tryGet("metadata", "heuristic_execution_time") / 60, 1)
        coordinator_execution_time = round(conf.tryGet("metadata", "coordinator_execution_time") / 60, 1)
        simulation_execution_time = round(conf.tryGet("metadata", "simulation_execution_time") / 60, 1)

        # Append the times to their respective lists
        values_x_axis.append(f"{value_x_axis} {type}")
        total_times.append(total_execution_time)
        heuristic_times.append(heuristic_execution_time)
        coordinator_times.append(coordinator_execution_time)
        simulation_times.append(simulation_execution_time)

    # Plotting
    plt.figure(figsize=(10,6))

    # Stacking each type of execution time (excluding total) on top of each other
    plt.bar(values_x_axis, heuristic_times, label='Heuristic')
    # Updating the bottom to stack the next layer
    new_bottom_heuristic = [heuristic for heuristic in heuristic_times]
    plt.bar(values_x_axis, coordinator_times, bottom=new_bottom_heuristic, label='Coordinator')
    # Repeat for the next layer
    new_bottom_coordinator = [heuristic + coordinator for heuristic, coordinator in zip(heuristic_times, coordinator_times)]
    plt.bar(values_x_axis, simulation_times, bottom=new_bottom_coordinator, label='Simulation')

    # Adding the total execution time as text above each bar
    for i, total in enumerate(total_times):
        plt.text(values_x_axis[i], total, str(total), ha='center', va='bottom')

    # Get the heuristic information from the path.
    dirs = Path(path).parts

    heuristic_type = dirs[-3]
    algorithm = dirs[-2]

    plt.title(f"Distinct execution times for {heuristic_type} - {algorithm} with increasing {type}.")
    plt.ylabel("Time (minutes)")
    plt.legend()

    # Saving the plot
    os.makedirs(os.path.dirname("/home/larry/hyper-heuristic-dse-2.0/data/processed/execution_times/"), exist_ok=True)
    plt.savefig(f"/home/larry/hyper-heuristic-dse-2.0/data/processed/execution_times/bars_{type}.png")


'''
    Plot the execution times for each file as a pie chart.

    The pie chart shows the relative execution times of the heuristic, coordinator and simulation.
'''
def combined_pie_execution_times(type="components"):
    path = "/home/larry/hyper-heuristic-dse-2.0/data/raw/results/metaheuristic/random_search/"

    # Get the list of files
    files = glob.glob(f"{path}*.json")
    files.sort()

    # Determine the number of subplots needed (one for each file)
    n = len(files)
    _, axs = plt.subplots(1, n, figsize=(n * 6, 6))

    for i, file in enumerate(files):
        # Get the path of the file
        path = Path(file)

        # Create a config object with the path of the file
        conf = Config(path, Path("/home/larry/hyper-heuristic-dse-2.0/logs/vizualization"), "execution_times")

        # Get the execution times from the config object
        total_execution_time = round(conf.tryGet("metadata", "total_execution_time") / 60, 4)
        heuristic_execution_time = round(conf.tryGet("metadata", "heuristic_execution_time") / 60, 4)
        coordinator_execution_time = round(conf.tryGet("metadata", "coordinator_execution_time") / 60, 4)
        simulation_execution_time = round(conf.tryGet("metadata", "simulation_execution_time") / 60, 4)

        if total_execution_time > 0:
            # Calculate the relative times
            heuristic_percentage = round((heuristic_execution_time / total_execution_time) * 100, 4)
            coordinator_percentage = round((coordinator_execution_time / total_execution_time) * 100, 4)
            simulation_percentage = round((simulation_execution_time / total_execution_time) * 100, 4)

            # Labels and sizes for the pie chart
            labels = 'Heuristic', 'Coordinator', 'Simulation'
            sizes = [heuristic_percentage, coordinator_percentage, simulation_percentage]

            # Plotting the pie chart in the subplot
            axs[i].pie(sizes, labels=labels, autopct='%1.4f%%', startangle=140)
            value_x_axis = conf.tryGet("nr_of_components") if type == "components" else conf.tryGet("nr_of_iterations")
            axs[i].set_title(f"{value_x_axis} {type}")
            axs[i].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Adjust the layout
    plt.tight_layout()

    # Save the combined plot
    os.makedirs(os.path.dirname("/home/larry/hyper-heuristic-dse-2.0/data/processed/execution_times/"), exist_ok=True)
    plt.savefig(f"/home/larry/hyper-heuristic-dse-2.0/data/processed/execution_times/combined_pie_charts_{type}.png")


def plot_best_fitness_per_file():
    path = "/home/larry/hyper-heuristic-dse-2.0/data/raw/results/metaheuristic/random_search/"

    # Ensure the directory for saving plots exists
    os.makedirs(os.path.dirname("/home/larry/hyper-heuristic-dse-2.0/data/processed/execution_times/"), exist_ok=True)

    # Iterate through all json files
    files = glob.glob(f"{path}*.json")
    files.sort()

    for file in files:
        # Get the path of the file
        path = Path(file)

        # Create a config object with the path of the file
        conf = Config(path, Path("/home/larry/hyper-heuristic-dse-2.0/logs/vizualization"), "execution_times")
        best_fitness_values = conf.tryGet("historical_data", "best_fitness_after_every_iteration")
        nr_of_components = conf.tryGet("nr_of_components")

        # Plotting for each file
        plt.figure()
        plt.plot(best_fitness_values)
        plt.title(f"Best Fitness Value After Each Iteration - {nr_of_components} components")
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness Value")

        # Make steps in y axis 10
        plt.yticks(np.arange(0, 350, step=50))

        # Save the plot for each file
        plt.savefig(f"/home/larry/hyper-heuristic-dse-2.0/data/processed/execution_times/{nr_of_components}_best_fitness.png")
        plt.close()


def plot_convergence_vs_components():
    path = "/home/larry/hyper-heuristic-dse-2.0/data/raw/results/metaheuristic/random_search/"

    # Lists to store convergence numbers and number of components
    convergence_numbers = []
    nr_of_components_list = []

    # Iterate through all json files
    files = glob.glob(f"{path}*.json")
    files.sort()

    for file in files:
        # Get the path of the file
        path = Path(file)

        # Create a config object with the path of the file
        conf = Config(path, Path("/home/larry/hyper-heuristic-dse-2.0/logs/vizualization"), "execution_times")

        # Extract the best fitness values and the final most optimal solution's fitness
        best_fitness_values = conf.tryGet("historical_data", "best_fitness_after_every_iteration")
        optimal_fitness = conf.tryGet("optimal_found_solution", "optimal_found_fitness")

        # Find the convergence number (iteration where the optimal fitness is first achieved)
        convergence_number = next((i for i, fitness in enumerate(best_fitness_values) if fitness == optimal_fitness), None)

        # Extract the number of components
        nr_of_components = conf.tryGet("nr_of_components")

        # Append to lists
        convergence_numbers.append(convergence_number)
        nr_of_components_list.append(nr_of_components)

    # Plotting
    plt.figure()
    plt.scatter(nr_of_components_list, convergence_numbers)
    plt.title("Convergence Number vs Number of Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Convergence Number")

    # Make steps in y axis 1 from 0 to 10.
    plt.yticks(np.arange(0, 10, step=1))

    plt.grid(True)

    # Save the plot
    os.makedirs(os.path.dirname("/home/larry/hyper-heuristic-dse-2.0/data/processed/execution_times/"), exist_ok=True)
    plt.savefig(f"/home/larry/hyper-heuristic-dse-2.0/data/processed/execution_times/convergence_vs_components.png")


'''
    Save the best fitness values at every iteration step as
    a plot for fitness over iteration step.

    hist_values: Historicl values of run metaheuristic
    store_path:  Path to store the plot.
'''
def save_simulation_fitness(hist_values, store_path, title, filename):

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
    plt.ylabel('Latency and cost evaluation')
    plt.xlim(0, len(fitness_values) - 1)
    plt.ylim(0, max(fitness_values) + 1)
    plt.xticks(np.arange(0, len(fitness_values), step=1))
    plt.grid()

    # Add the formula as text annotation in the top right corner
    formula_text = f"(0.5 * (1 / end_to_end delay))\n + (0.5 * network_cost)"
    plt.text(0.5, 0.9, formula_text, fontsize=12, color='red', transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white', edgecolor='red'))

    # Add a title to your plot
    plt.title(title)

    plt.savefig(f"/home/larry/hyper-heuristic-dse-2.0/data/processed/{filename}.png")


'''
    Read the csv file in path /home/larry/hyper-heuristic-dse-2.0/src/external/simulation_model/workflow/results/ab5f71b5e22b1928ce6aeae0ae2aaadb/tictoc.csv to a dataframe.

    Define the dataframe such that it contains all rows with column 'type' having value 'statistic'. Skip on columns 'underflows,overflows,binedges,binvalues'.
'''
def save_hop_count():
    df = pd.read_csv("/home/larry/hyper-heuristic-dse-2.0/src/external/simulation_model/workflow/results/ab5f71b5e22b1928ce6aeae0ae2aaadb/tictoc.csv")
    df = df[df.type == 'statistic'].drop(columns=['underflows','overflows','binedges','binvalues'])

    # For every row, make a box plot with the values 'mean, stddev, min, max'. The plot title should be the value of column 'module'
    for index, row in df.iterrows():
        plt.figure()
        plt.boxplot([row['mean'], row['stddev'], row['min'], row['max']])
        plt.title(f'Hop count distribution - {row["module"]}')
        plt.savefig("/home/larry/hyper-heuristic-dse-2.0/data/processed/" + row['module'] + ".png")


def save_average_response_time():
    # # Specify the directory to add to PATH
    # new_path = '/home/larry/hyper-heuristic-dse-2.0/omnetpp-6.0.1/bin'

    # # Get the current PATH variable
    # current_path = os.environ.get('PATH', '')

    # # Add the new directory to PATH if it's not already present
    # if new_path not in current_path:
    #     os.environ['PATH'] = new_path + os.pathsep + current_path

    # command = "opp_scavetool export -F CSV-R -o x.csv *.sca"
    # directory = r"/home/larry/hyper-heuristic-dse-2.0/src/external/simulation_model/workflow/results/343ddead4455987aae4c6e56e3cb132a"

    # try:
    #     result = subprocess.run(command, shell=True, cwd=directory, check=True)
    #     print("Command executed successfully.")
    # except subprocess.CalledProcessError as e:
    #     print(f"Error executing the command: {e}")

    # Load the CSV file "x.csv" generated by opp_scavetool
    csv_file_path = "/home/larry/hyper-heuristic-dse-2.0/src/external/simulation_model/workflow/results/343ddead4455987aae4c6e56e3cb132a/x.csv"

    # Load the CSV file "x.csv" generated by opp_scavetool with specified data types
    df = pd.read_csv(csv_file_path)

    # Filter rows with 'type' column equal to 'statistic' and drop unnecessary columns
    df = df[df['type'] == 'statistic'].drop(columns=['underflows', 'overflows', 'binedges', 'binvalues'])

    # Drop rows with 'NA' values
    df = df.dropna()

    # For every row in the filtered DataFrame, create a box plot
    for index, row in df.iterrows():
        plt.figure()
        data = [row['mean'], row['stddev'], row['min'], row['max']]
        plt.boxplot(data)
        plt.title(f'Average response time - {row["run"]}')

        # Save the plot as an image with the module name as the filename
        image_path = f"/home/larry/hyper-heuristic-dse-2.0/data/processed/{row['run']}.png"
        plt.savefig(image_path)
        plt.close()  # Close the plot to release resources

    # # Optional: Show or save a summary plot of all the generated box plots
    # # You can adjust the layout and appearance as needed
    # plt.figure()
    # plt.boxplot(df[['mean', 'stddev', 'min', 'max']].values.T)
    # plt.title('Summary of Average response time')
    # plt.xticks(range(1, len(df) + 1), df['run'], rotation=45)
    # plt.savefig("/home/larry/hyper-heuristic-dse-2.0/data/processed/summary.png")
    # plt.close()  # Close the summary plot

    # print("Box plots generated and saved.")


if __name__ == "__main__":
    pass
