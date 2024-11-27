import json
from math import inf
import os
import glob
from matplotlib import axis
import numpy as np
import pandas as pd
import math 
import datetime
import scipy.stats as st
import random
import matplotlib.colors as mcolors


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from pathlib import Path
import matplotlib.pyplot as plt


class visuSteps:
    def __init__(self, directory_path, result_dir=None):
        self.df = None
        self.rawdf = None
        self.directory_path = directory_path
        if result_dir is None:
            self.result_dir = directory_path
        else:
            self.result_dir = result_dir
        self.conf = self.getConfig()
    
    def getConfig(self):
        filepath = Path(os.path.join(self.directory_path, "config.json"))
        with open(filepath, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            dataf = data["file_details"]
            do_list = list(str(x["param_name"])+": "+str(x["configuration_pattern"][0]) for x in dataf["coordinator_config"]["simulation_model"]["simulation_model_params"])
            do_string = ' // '.join(str(x) for x in do_list)
            return {
                "data": data,
                "time": datetime.fromtimestamp(dataf["timestamp"]).strftime("%Y-%m-%d %H:%M"),
                "server":   "Platform: "+str(dataf["coordinator_config"]["simulation_model"]["simulation_model_configuration"]["platform"]) + \
                            " ("+str(dataf["coordinator_config"]["simulation_model"]["simulation_model_configuration"]["jobs"]) + " Jobs, " + \
                                str(dataf["coordinator_config"]["simulation_model"]["simulation_model_configuration"]["job_cores"]) + " " +\
                                str(dataf["coordinator_config"]["simulation_model"]["simulation_model_configuration"]["job_memory"]) + "GB) ",
                "designobjectives":  do_string,
                "hh":   "Search operator: "+str(dataf["search_operator_space_name"]) + \
                        "\nSteps: "+str(data["paramaters"]["num_steps"]) + \
                        " | Iterations: "+str(data["paramaters"]["num_iterations"]) + \
                        " | Agents: "+str(data["paramaters"]["num_agents"]) + \
                        " | Replicas: "+str(data["paramaters"]["num_replicas"]) + \
                        "\nFitFunc: "+str(dataf["coordinator_config"]["fitness_config"]["fitness_function"]),
                "hh_small":   str(dataf["search_operator_space_name"]) + \
                        "\nSteps: "+str(data["paramaters"]["num_steps"]) + \
                        " | Iterations: "+str(data["paramaters"]["num_iterations"]) + \
                        " | Agents: "+str(data["paramaters"]["num_agents"]) + \
                        " | Replicas: "+str(data["paramaters"]["num_replicas"]) 
            }
    def get_statistics(self, raw_data):
            # Get descriptive statistics
            with np.errstate(divide='ignore', invalid='ignore'):
                dst = st.describe(raw_data, nan_policy='omit')

            # Store statistics
            return dict(nob=dst.nobs,
                        Min=dst.minmax[0],
                        Max=dst.minmax[1],
                        Avg=dst.mean,
                        Std=np.std(raw_data),
                        Skw=dst.skewness,
                        Kur=dst.kurtosis,
                        IQR=st.iqr(raw_data),
                        Med=np.median(raw_data),
                        MAD=st.median_abs_deviation(raw_data))
        
    def plot_hh_multiplot(self):
        all_historical_fitness = []
        mh = self.conf["data"]["search_operator_space_name"]

        # List all files in the directory
        for filename in os.listdir(self.directory_path):
            # Construct full file path
            file_path = os.path.join(self.directory_path, filename)

            # Check if the file is a JSON file
            if filename.endswith('.json') and os.path.isfile(file_path) and not filename.endswith('config.json'):
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    try:
                        # Load JSON content
                        data = json.load(json_file)
                        rep_values = {}
                        for replica in data['details']['historical']:
                            # print(replica['fitness'])
                            for i in range(len(replica['fitness'])):
                                if not i in rep_values:
                                    rep_values[i] = []
                                rep_values[i].append(replica['fitness'][i])
                            i_rap = len(replica['fitness'])
                        historical_values_stats = {
                            "Med": [],
                            "Min": [],
                            "Max": []
                        }
                        for i in range(i_rap):
                            stats =  self.get_statistics(rep_values[i])
                            historical_values_stats["Med"].append(stats["Med"])
                            historical_values_stats["Min"].append(stats["Med"]-(stats["IQR"]/2))
                            historical_values_stats["Max"].append(stats["Med"]+(stats["IQR"]/2))
                        historical_fitness = historical_values_stats
                        all_historical_fitness.append(historical_fitness)
                    except json.JSONDecodeError as e:
                        print(f"Error reading {file_path}: {e}")

        num_subplots = math.ceil(len(all_historical_fitness)/3 + 0.001) + 1
        fig, axs = plt.subplots(num_subplots, 1, figsize=(10, 6*num_subplots))
        fig.text(0.01,0.975, self.conf["hh"])
        fig.text(0.875,0.975, self.conf["time"])
        fig.text(0.01,0.005, self.conf["server"])
    # Sort colors by hue, saturation, value and name.

        names = list(mcolors.CSS4_COLORS)
        random.Random(69).shuffle(names)
        names = list(mcolors.TABLEAU_COLORS) + names

        for idx, fitness_values in enumerate(all_historical_fitness):
            ids = math.ceil((idx/3) + 0.001)
            # print(idx, ids/3,  math.ceil(idx/3),  math.ceil(idx/3) + 1)
            axs[0].plot(fitness_values['Med'], label=f'HH Step: {idx}', color=names[idx])
            axs[ids].plot(fitness_values['Med'], label=f'HH Step: {idx}', color=names[idx])
            axs[ids].plot(fitness_values['Min'], color=names[idx], linestyle = 'dotted')
            axs[ids].plot(fitness_values['Max'], color=names[idx], linestyle = 'dotted')


        for i, ax in enumerate(axs):
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Fitness')
            # axs[1].set_title(f'Optimal Fitness Every Iteration with Min/Max in dotted\n{mh}')
            ax.legend()

        axs[0].set_xlabel('Iteration')
        axs[0].set_ylabel('Fitness')
        axs[0].set_title(f'Optimal Fitness Every Iteration\n{mh}')
        axs[0].legend()

        # Save the plot in the same directory
        plot_file_path = os.path.join(self.result_dir, mh+'_historical_fitness_plot.png')

        fig.tight_layout()
        fig.savefig(plot_file_path)

        print(f"plot_hh_multiplot: Plot saved as {plot_file_path}")
    
    def plot_hh_boxplot(self):
        mh = self.conf["data"]["search_operator_space_name"]

        num_files = 0
        for filename in os.listdir(self.directory_path):
            file_path = os.path.join(self.directory_path, filename)
            if filename.endswith('.json') and os.path.isfile(file_path) and not filename.endswith('config.json'):
                num_files += 1

        fig, axs = plt.subplots(num_files, 1, figsize=(10, 6*num_files))
        names = list(mcolors.TABLEAU_COLORS)
        # List all files in the directory
        i_step = 0
        iteration_value_step_list = []
        for filename in os.listdir(self.directory_path):
            # Construct full file path
            file_path = os.path.join(self.directory_path, filename)

            # Check if the file is a JSON file
            if filename.endswith('.json') and os.path.isfile(file_path) and not filename.endswith('config.json'):
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    try:
                        # Load JSON content
                        data = json.load(json_file)
                        iteration_value_step = {}
                        for replica in data['details']['historical']:
                            for i in range(len(replica['fitness'])):
                                if not i in iteration_value_step:
                                    iteration_value_step[i] = []
                                iteration_value_step[i].append(replica['fitness'][i])
                        iteration_value_step_list.append(list(iteration_value_step.values()))
                        i_step +=1
                        
                    except json.JSONDecodeError as e:
                        print(f"Error reading {file_path}: {e}")

    # Sort colors by hue, saturation, value and name.
        result = []
        for inner_list in iteration_value_step_list:
            for inner in inner_list:
                result.append(min(inner))
                result.append(max(inner))
        ylim = [min(result), max(result)]


        for i in range(i_step):
            axs[i].boxplot(iteration_value_step_list[i], showmeans=True, meanline=True)
            axs[i].set_title('HH Step: '+str(i))
            axs[i].set_ylim(ylim)

        # Save the plot in the same directory
        plot_file_path = os.path.join(self.result_dir, mh+'_historical_fitness_boxplot.png')
        fig.text(0.01,0.98, self.conf["hh"])
        fig.text(0.875,0.98, self.conf["time"])
        fig.text(0.01,0.005, self.conf["server"])
        # fig.suptitle(mh)
        axs[0].set_title(f'\nHH Step: 0')
        fig.tight_layout()
        fig.savefig(plot_file_path)

        print(f"plot_hh_boxplot: Plot saved as {plot_file_path}")

    def plot_hh_boxplot_all(self):
        iteration_values = {}
        mh = self.conf["data"]["search_operator_space_name"]

        # List all files in the directory
        for filename in os.listdir(self.directory_path):
            # Construct full file path
            file_path = os.path.join(self.directory_path, filename)

            # Check if the file is a JSON file
            if filename.endswith('.json') and os.path.isfile(file_path) and not filename.endswith('config.json'):
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    try:
                        # Load JSON content
                        data = json.load(json_file)
                        for replica in data['details']['historical']:
                            for i in range(len(replica['fitness'])):
                                if not i in iteration_values:
                                    iteration_values[i] = []
                                iteration_values[i].append(replica['fitness'][i])
                    except json.JSONDecodeError as e:
                        print(f"Error reading {file_path}: {e}")

        iteration_values_list = list(iteration_values.values())

        plt.figure(figsize=(10, 6))

        i_col = 0
        plt.boxplot(iteration_values_list, showmeans=True)
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.title(f'Boxplot of all HH steps combined\n{self.conf["hh_small"]} | {self.conf["time"]}')

        plt.legend()

        # Save the plot in the same directory
        plot_file_path = os.path.join(self.result_dir, mh+'_historical_fitness_boxplot_all.png')
        plt.savefig(plot_file_path)
        plt.close()

        print(f"plot_hh_boxplot_all: Plot saved as {plot_file_path}")

    def plot(self):
        self.plot_hh_multiplot()
        self.plot_hh_boxplot()
        self.plot_hh_boxplot_all()


if __name__ == "__main__":
    ps = visuSteps("", "")

    ps.plot()
