import os
import numpy as np
import pandas as pd
import glob

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


'''
    Deprecated: Please use new_function instead.
'''
def prepare_fitness_data(hist_values):
    # Direct extraction of 'fitness' values
    fitness_values = hist_values['fitness']
    return fitness_values


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


'''
    Deprecated: Please use new_function instead.
'''
def plot_fitness_over_iterations(fitness_values, store_path, title, formula_text):
    plt.figure()
    plt.plot(fitness_values, lw=2)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Evaluation')
    plt.xlim(0, len(fitness_values) - 1)
    plt.ylim(0, max(fitness_values) + 1)
    plt.xticks(np.arange(0, len(fitness_values), step=1))
    plt.grid()

    # Add the formula as text annotation in the top right corner
    plt.text(0.5, 0.9, formula_text, fontsize=12, color='red', transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white', edgecolor='red'))

    plt.title(title)
    plt.savefig(store_path)
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


def main_plot_convergence_vs_components(data_path, visualization_path, output_path):
    convergence_numbers = []
    nr_of_components_list = []

    for config_path in sorted(data_path.glob('*.json')):
        convergence_number, nr_of_components = get_convergence_data(config_path, visualization_path)
        if convergence_number is not None:  # Ensure valid convergence was found
            convergence_numbers.append(convergence_number)
            nr_of_components_list.append(nr_of_components)

    plot_convergence(convergence_numbers, nr_of_components_list, output_path)


'''
    Deprecated: Please use new_function instead.
'''
def save_simulation_fitness(hist_values, store_path, title, filename):
    # Ensure the directory exists
    output_path = Path(store_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data
    fitness_values = prepare_fitness_data(hist_values)

    # Define formula text for annotation
    formula_text = "(0.5 * (1 / end_to_end delay)) + (0.5 * network_cost)"

    # Complete output path including filename
    complete_store_path = output_path / f"{filename}.png"

    # Plot and save the figure
    plot_fitness_over_iterations(fitness_values, complete_store_path, title, formula_text)


if __name__ == "__main__":
    # Example usage
    base_path = Path.cwd()
    data_path = base_path / "data/raw/results/metaheuristic/genetic_algorithm"
    visualization_path = Path("/home/lvdwater/hyper-heuristic-dse-2.0/logs/vizualization")
    output_path_bars = Path("/home/lvdwater/hyper-heuristic-dse-2.0/data/processed/execution_times/bars_components.png")
    output_path_pies = Path("/home/lvdwater/hyper-heuristic-dse-2.0/data/processed/execution_times/pies_components.png")
    plot_output_base = Path("/home/lvdwater/hyper-heuristic-dse-2.0/data/processed/execution_times/fitness_plots")
    output_path_convergence = Path("/home/lvdwater/hyper-heuristic-dse-2.0/data/processed/execution_times/convergence_vs_components.png")

    bar_execution_times(data_path, visualization_path, output_path_bars)
    combined_pie_execution_times(data_path, visualization_path, output_path_pies)
    main_plot_fitness_across_files(data_path, visualization_path, plot_output_base)
    main_plot_convergence_vs_components(data_path, visualization_path, output_path_convergence)
