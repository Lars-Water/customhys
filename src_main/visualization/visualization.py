import os
import numpy as np
import pandas as pd

from src_main.tools.config_reader import Config
from pathlib import Path

import matplotlib.pyplot as plt
plt.rcParams.update({'text.usetex': False})


# Data extraction functions


def get_execution_times_from_config(config_path, visualization_path, type):
    os.makedirs(visualization_path, exist_ok=True)
    conf = Config(config_path, visualization_path, "execution_times")
    metrics = ["nr_of_components", "nr_of_iterations", "metadata"]
    keys = ["nr_of_components" if type == "components" else "nr_of_iterations", "total_execution_time",
            "heuristic_execution_time", "coordinator_execution_time", "simulation_execution_time"]
    values = [conf.tryGet(metric) if metric in metrics else conf.tryGet("metadata", metric) for metric in keys]
    return [round(value / 60, 1) if "time" in key else value for key, value in zip(keys, values)]


def get_fitness_values_from_config(config_path, visualization_path):
    os.makedirs(visualization_path, exist_ok=True)
    conf = Config(config_path, visualization_path, "execution_times")
    best_fitness_values = conf.tryGet("historical_data", "best_fitness_after_every_iteration")
    nr_of_components = conf.tryGet("nr_of_components")
    return best_fitness_values, nr_of_components


def get_convergence_data(config_path, visualization_path):
    os.makedirs(visualization_path, exist_ok=True)
    conf = Config(config_path, visualization_path, "execution_times")
    best_fitness_values = conf.tryGet("historical_data", "best_fitness_after_every_iteration")
    optimal_fitness = conf.tryGet("optimal_found_solution", "optimal_found_fitness")
    nr_of_components = conf.tryGet("nr_of_components")

    convergence_number = next((i for i, fitness in enumerate(best_fitness_values) if fitness == optimal_fitness), None)

    return convergence_number, nr_of_components


def find_non_dominated_points(df, x_col, y_col):
    # Sort points by x descending (for latency maximization) and y ascending (for cost minimization)
    sorted_points = df.sort_values(by=[x_col, y_col], ascending=[False, True])

    # Initialize the list of non-dominated points
    non_dominated = []

    # Initialize the best known y-value (minimization)
    best_y = float('inf')

    # Traverse the points
    for index, row in sorted_points.iterrows():
        if row[y_col] < best_y:
            non_dominated.append(index)
            best_y = row[y_col]

    return non_dominated


# Plotting functions


def plot_execution_times(values_x_axis, execution_times, plot_title, output_path):
    plt.figure(figsize=(10, 6))
    labels = ['Heuristic', 'Coordinator', 'Simulation']
    bottom = np.zeros(len(values_x_axis))

    for times, label in zip(execution_times[:-1], labels):  # Exclude total times from stacking
        plt.bar(values_x_axis, times, label=label, bottom=bottom)
        bottom += np.array(times)

    # Add total execution times as text above bars
    for idx, total in enumerate(execution_times[-1]):
        plt.text(idx, total, str(total), ha='center', va='bottom')

    plt.title(plot_title)
    plt.ylabel("Time (minutes)")
    plt.legend()

    # Fit y-axis to the size of the highest bar
    plt.ylim(0, max(sum(execution_times[:-1], [])) + max(execution_times[-1]))

    plt.savefig(output_path)


def plot_pie_chart(ax, execution_times, title):
    labels = ['Heuristic', 'Coordinator', 'Simulation']
    sizes = execution_times  # Assuming execution_times are already in percentage
    colors = ['gold', 'lightcoral', 'lightskyblue']  # Optional: Define colors for each slice for consistency

    # Plotting the pie chart
    wedges, _, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)

    ax.set_title(title)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Improve legibility of autopct labels on small slices
    for autotext in autotexts:
        autotext.set_color('white')  # Change color to white (or any contrasting color)

    # Adding legend
    ax.legend(wedges, labels, title="Components", loc="best", bbox_to_anchor=(1, 0, 0.5, 1))


def plot_best_fitness(fitness_values, nr_of_components, plot_path):
    plt.figure()
    plt.plot(fitness_values)
    plt.title(f"Best Fitness Value After Each Iteration - {nr_of_components} components")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness Value")
    plt.yticks(np.arange(0, 350, step=50))
    plt.savefig(plot_path)
    plt.close()


def plot_convergence(convergence_numbers, nr_of_components_list, output_path):
    plt.figure()
    plt.scatter(nr_of_components_list, convergence_numbers)
    plt.title("Convergence Number vs Number of Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Convergence Number")
    plt.yticks(np.arange(min(convergence_numbers), max(convergence_numbers) + 1, step=1))
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def plot_convergence_against_iterations(best_fitness_after_every_iteration, output_path):
    plt.figure()
    plt.plot(best_fitness_after_every_iteration)
    plt.title("Best Fitness Value After Each Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness Value")
    plt.yticks(np.arange(0, 1.5, step=0.1))
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def quick_and_dirty_multiplot():
    plt.figure()

    output_path = Path("/home/larry/hyper-heuristic-dse-2.0/data/raw/results/metaheuristic/pso_swarm_conf")
    data1 = pd.read_csv(Path("/home/larry/hyper-heuristic-dse-2.0/data/raw/results/metaheuristic/pso_swarm_conf/default/1716914576/convergence.csv"))
    data1 = data1['best_fitness_value_untill_current_iteration'].values
    plt.plot(data1, label="Default - PSO - Swarm Conf: 2.5")
    data2 = pd.read_csv(Path("/home/larry/hyper-heuristic-dse-2.0/data/raw/results/metaheuristic/pso_swarm_conf/pso_tune_swarm_conf_0_01/1717079252/convergence.csv"))
    data2 = data2['best_fitness_value_untill_current_iteration'].values
    plt.plot(data2, label="PSO - Swarm Conf: 0.01")
    data3 = pd.read_csv(Path("/home/larry/hyper-heuristic-dse-2.0/data/raw/results/metaheuristic/pso_swarm_conf/pso_tune_swarm_conf_4_99/1717076328/convergence.csv"))
    data3 = data3['best_fitness_value_untill_current_iteration'].values
    plt.plot(data3, label="PSO - Swarm Conf: 4.99")
    # data4 = pd.read_csv(Path("/home/larry/hyper-heuristic-dse-2.0/data/raw/results/metaheuristic/ga_genmut_mutrate/ga_tune_genmut_mutrate_0_6/1717117393/convergence.csv"))
    # data4 = data4['best_fitness_value_untill_current_iteration'].values
    # plt.plot(data4, label="Mutation Rate: 0.6")
    # data5 = pd.read_csv(Path("/home/larry/hyper-heuristic-dse-2.0/data/raw/results/metaheuristic/ga_genmut_mutrate/ga_tune_genmut_mutrate_0_9/1717120764/convergence.csv"))
    # data5 = data5['best_fitness_value_untill_current_iteration'].values
    # plt.plot(data5, label="Mutation Rate: 0.9")

    plt.title("Best Fitness Value After Each Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness Value")
    plt.yticks(np.arange(0, 1.0, step=0.1))
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)
    plt.close()


# Main functions


def bar_execution_times(data_path, visualization_path, output_path, type="components"):
    # Initialize lists to store execution times
    execution_times = [[] for _ in range(5)]  # heuristic, coordinator, simulation, total, values_x_axis
    label_suffix = type

    # Iterate through all json files
    for config_path in sorted(data_path.glob("*.json")):
        times = get_execution_times_from_config(config_path, visualization_path, type)
        execution_times[0].append(times[2])  # heuristic
        execution_times[1].append(times[3])  # coordinator
        execution_times[2].append(times[4])  # simulation
        execution_times[3].append(times[1])  # total
        execution_times[4].append(f"{times[0]} {label_suffix}")

    plot_title = f"Distinct execution times with increasing {type}."
    plot_execution_times(execution_times[4], execution_times[:4], plot_title, output_path)


def combined_pie_execution_times(data_path, visualization_path, output_path, type="components"):
    files = sorted(data_path.glob("*.json"))
    n = len(files)
    _, axs = plt.subplots(1, n, figsize=(n * 6, 6))

    for i, config_path in enumerate(files):
        times = get_execution_times_from_config(config_path, visualization_path, type)
        # Calculate percentages
        total_time = times[1]  # Assuming this is the total execution time
        percentages = [time / total_time * 100 for time in times[2:5]]  # heuristic, coordinator, simulation times

        value_x_axis = times[0]  # Assuming this is nr_of_components or nr_of_iterations
        title = f"{value_x_axis} {type}"

        plot_pie_chart(axs[i], percentages, title)

    plt.tight_layout()
    plt.savefig(output_path)


def main_plot_fitness_across_files(data_path, visualization_path, plot_output_base):
    paths = list(data_path.glob('*.json'))  # Consider using Path objects from pathlib
    os.makedirs(plot_output_base, exist_ok=True)

    for config_path in sorted(paths):
        fitness_values, nr_of_components = get_fitness_values_from_config(config_path, visualization_path)
        plot_path = plot_output_base / f"{nr_of_components}_best_fitness.png"
        plot_best_fitness(fitness_values, nr_of_components, plot_path)


def main_plot_convergence_csv(convergence_data_path, output_path):
    convergence_data = pd.read_csv(convergence_data_path)
    best_fitness_after_every_iteration = convergence_data['best_fitness_value_untill_current_iteration'].values
    plot_convergence_against_iterations(best_fitness_after_every_iteration, output_path)


def main_plot_convergence_vs_components(data_path, visualization_path, output_path):
    convergence_numbers = []
    nr_of_components_list = []

    for config_path in sorted(data_path.glob('*.json')):
        convergence_number, nr_of_components = get_convergence_data(config_path, visualization_path)
        if convergence_number is not None:  # Ensure valid convergence was found
            convergence_numbers.append(convergence_number)
            nr_of_components_list.append(nr_of_components)

    plot_convergence(convergence_numbers, nr_of_components_list, output_path)


def main_plot_design_space(temp_design_points_data_file, output_path_design_space, nr_of_operators_mh=1):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path_design_space), exist_ok=True)

    # Read the design space data with initial skip
    reader = pd.read_csv(temp_design_points_data_file, iterator=True)

    design_space_data = pd.DataFrame()

    if nr_of_operators_mh == 1:
        # Save the remaining rows in a dataframe.
        design_space_data = reader.get_chunk()

    elif nr_of_operators_mh > 1:
        # Read the first 16 rows.
        design_space_data = reader.get_chunk(16)
        while True:
            # Skip the next 16 rows.
            try:
                reader.get_chunk(16)
            except StopIteration:
                break
            # Read the next 16 rows.
            design_space_data = pd.concat([design_space_data] + [reader.get_chunk(16) for _ in range(nr_of_operators_mh - 1)])

    # Close the iterator to release resources
    reader.close()

    # Identify non-dominated points
    non_dominated_indices = find_non_dominated_points(design_space_data, 'AdjustedLatency', 'AdjustedNetworkCost')

    # Plot the design space data
    plt.figure()
    x_coordinates = design_space_data['AdjustedLatency']
    y_coordinates = design_space_data['AdjustedNetworkCost']

    # Plot the dominated solutions.
    plt.scatter(x_coordinates, y_coordinates, alpha=0.5, label='Dominated Points')
    # Highlight non-dominated points in red.
    plt.scatter(x_coordinates[non_dominated_indices],
                y_coordinates[non_dominated_indices],
                color='red', label='Non-dominated Points')

    plt.title("Design Points in the Design Space")
    plt.xlabel("Latency * 0.5")
    plt.ylabel("Network Cost * 0.5")

    # Set the limits of x and y axes
    x_min = min(x_coordinates)
    x_max = max(x_coordinates)
    y_min = min(y_coordinates)
    y_max = max(y_coordinates)
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)

    # Add a legend
    plt.legend()

    plt.savefig(output_path_design_space)
    plt.close()


if __name__ == "__main__":
    # # Example usage
    # base_path = Path.cwd()
    # data_path = base_path / "data/raw/results/metaheuristic/genetic_algorithm"
    # visualization_path = Path("/home/lvdwater/hyper-heuristic-dse-2.0/logs/vizualization")
    # output_path_bars = Path("/home/lvdwater/hyper-heuristic-dse-2.0/data/processed/execution_times/bars_components.png")
    # output_path_pies = Path("/home/lvdwater/hyper-heuristic-dse-2.0/data/processed/execution_times/pies_components.png")
    # plot_output_base = Path("/home/lvdwater/hyper-heuristic-dse-2.0/data/processed/execution_times/fitness_plots")
    # output_path_convergence = Path("/home/lvdwater/hyper-heuristic-dse-2.0/data/processed/execution_times/convergence_vs_components.png")

    # bar_execution_times(data_path, visualization_path, output_path_bars)
    # combined_pie_execution_times(data_path, visualization_path, output_path_pies)
    # main_plot_fitness_across_files(data_path, visualization_path, plot_output_base)
    # main_plot_convergence_vs_components(data_path, visualization_path, output_path_convergence)

    # temp_design_points_data_file = Path("/home/larry/hyper-heuristic-dse-2.0/data/raw/design_points/design_point_metrics_3_bb_switches.csv")
    # output_path_design_space = Path("/home/larry/hyper-heuristic-dse-2.0/data/processed/design_space/design_space_3_bb_switches.png")
    # main_plot_design_space(temp_design_points_data_file, output_path_design_space)
    pass
