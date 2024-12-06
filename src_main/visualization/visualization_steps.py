import json
from math import inf
import os
import glob
from matplotlib import axis
import numpy as np
import pandas as pd
import math 
from datetime import datetime
import scipy.stats as st
import random
import matplotlib.colors as mcolors


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from pathlib import Path
import matplotlib.pyplot as plt
from visualization_base import visuBase


class visuSteps(visuBase):
    def __init__(self, directory_path, result_dir=None, reason_dir=None):
        super().__init__()
        self.reason_dir = reason_dir
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
                "time": datetime.fromtimestamp(dataf["timestamp_int"]).strftime("%Y-%m-%d %H:%M"),
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
        
    def plot_hh_multiplot(self, plot_file_path=None):
        all_historical_fitness = []
        mh = self.conf["data"]["file_details"]["search_operator_space_name"]

        # List all files in the directory
        for filename in os.listdir(self.directory_path):
            # Construct full file path
            file_path = os.path.join(self.directory_path, filename)

            # Check if the file is a JSON file
            if filename.endswith('.json') and os.path.isfile(file_path) and not filename.endswith('config.json') and not filename.endswith("solutions.json"):
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    try:
                        # Load JSON content
                        data = json.load(json_file)
                        rep_values = {}
                        if type(data["candidate"]['details']) == dict:
                            for replica in data["candidate"]['details']['historical']:
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
        if plot_file_path == None:
            plot_file_path = os.path.join(self.result_dir, mh+'_historical_fitness_plot.png')
        else: 
            plot_file_path = plot_file_path

        fig.tight_layout()
        fig.savefig(plot_file_path)

        print(f"plot_hh_multiplot: Plot saved as {plot_file_path}")
    
    def plot_hh_boxplot(self, plot_file_path=None):
        mh = self.conf["data"]["file_details"]["search_operator_space_name"]

        num_files = 0
        for filename in os.listdir(self.directory_path):
            file_path = os.path.join(self.directory_path, filename)
            if filename.endswith('.json') and os.path.isfile(file_path) and not filename.endswith('config.json') and not filename.endswith("solutions.json"):
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
            if filename.endswith('.json') and os.path.isfile(file_path) and not filename.endswith('config.json') and not filename.endswith("solutions.json"):
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    try:
                        # Load JSON content
                        data = json.load(json_file)
                        iteration_value_step = {}
                        if type(data["candidate"]['details']) == dict:
                            for replica in data["candidate"]['details']['historical']:
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
        # print(iteration_value_step_list)
        for inner_list in iteration_value_step_list:
            for inner in inner_list:
                result.append(min(inner))
                result.append(max(inner))
        # print(result)
        ylim = [min(result), max(result)]


        for i in range(i_step):
            axs[i].boxplot(iteration_value_step_list[i], showmeans=True, meanline=True)
            axs[i].set_title('HH Step: '+str(i))
            axs[i].set_ylim(ylim)

        # Save the plot in the same directory
        if plot_file_path == None:
            plot_file_path = os.path.join(self.result_dir, mh+'_historical_fitness_boxplot.png')        
        else: 
            plot_file_path = plot_file_path

        fig.text(0.01,0.98, self.conf["hh"])
        fig.text(0.875,0.98, self.conf["time"])
        fig.text(0.01,0.005, self.conf["server"])
        # fig.suptitle(mh)
        axs[0].set_title(f'\nHH Step: 0')
        fig.tight_layout()
        fig.savefig(plot_file_path)

        print(f"plot_hh_boxplot: Plot saved as {plot_file_path}")

    def plot_hh_boxplot_all(self, plot_file_path=None):
        iteration_values = {}
        mh = self.conf["data"]["file_details"]["search_operator_space_name"]

        # List all files in the directory
        for filename in os.listdir(self.directory_path):
            # Construct full file path
            file_path = os.path.join(self.directory_path, filename)

            # Check if the file is a JSON file
            if filename.endswith('.json') and os.path.isfile(file_path) and not filename.endswith('config.json') and not filename.endswith("solutions.json"):
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    try:
                        # Load JSON content
                        data = json.load(json_file)
                        if type(data["candidate"]['details']) == dict:
                            for replica in data["candidate"]['details']['historical']:
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
        if plot_file_path == None:
            plot_file_path = os.path.join(self.result_dir, mh+'_historical_fitness_boxplot_all.png')
        else: 
            plot_file_path = plot_file_path

        plt.savefig(plot_file_path)
        plt.close()

        print(f"plot_hh_boxplot_all: Plot saved as {plot_file_path}")

    def _plt_title_tex(self, title, info, df = None):
        if df is None:
            df = self.df
        title_tex = '{\\fontsize{20pt}{3em}\\selectfont{}'+title+'}'
        if info is not None:
            title_tex += ' \n{\\fontsize{8pt}{3em}\\selectfont{}'+str(info)+'}'
        return title_tex
    
    def plot_line_best(self, 
                       reason_dir, 
                       plot_file_path=None,  
                       figsize=(6, 4),
                       info = None
                    ):
        x_axis_name = "Step"
        y_axis_name = "Fitness"
        x_axis_name, y_axis_name = self._plt_axis_names(x_axis_name, y_axis_name)
                     
        title = "Performance"
        plt, x_axis_name, y_axis_name = self._plt(title, x_axis_name, y_axis_name, x_axis_name, y_axis_name, figsize, info)

        reasons = self.readDynamicSelectionReasons(reason_dir)
        ymax = 0
        names = list(mcolors.TABLEAU_COLORS)
        i = 0
        for search_operator, reason in reasons.items():
            df = pd.DataFrame(reason["steps"]).transpose()
            plt.plot(df["performance"], color=names[i], linestyle = 'dotted')
            plt.plot(df["best"], label=f"{search_operator}", color=names[i])
            ymax = max(ymax, max(df["performance"]))
            ymax = max(ymax, max(df["best"]))
            i += 1

        self.plot_Reasons(reason_dir, plt, ymax)

        plt.legend()
        plt.tight_layout()
        # plt.show()
        if plot_file_path == None:
            plot_file_path = os.path.join(self.result_dir, 'hh_performance_line.png')
        else: 
            plot_file_path = plot_file_path
        plt.savefig(plot_file_path, dpi=150)
        print(f"plot_line_best: Plot saved as {plot_file_path}")
    
    def plot_agents(self, 
                       reason_dir, 
                       plot_file_path=None,  
                       figsize=(6, 4),
                       info = None
                    ):
        x_axis_name = "Step"
        y_axis_name = "Fitness"
        x_axis_name, y_axis_name = self._plt_axis_names(x_axis_name, y_axis_name)
                     
        title = "Performance"
        plt, x_axis_name, y_axis_name = self._plt(title, x_axis_name, y_axis_name, x_axis_name, y_axis_name, figsize, info)

        reasons = self.readDynamicSelectionReasons(reason_dir)
        print(reasons)
        agents_nums = {}
        for search_operator, reason in reasons.items():
            agents_nums[search_operator] = {
                    
            }
        ymax = 0
        names = list(mcolors.TABLEAU_COLORS)
        i = 0
        for search_operator, reason in reasons.items():
            df = pd.DataFrame(reason["steps"]).transpose()
            plt.plot(df["performance"], color=names[i], linestyle = 'dotted')
            plt.plot(df["best"], label=f"{search_operator}", color=names[i])
            ymax = max(ymax, max(df["performance"]))
            ymax = max(ymax, max(df["best"]))
            i += 1

        self.plot_Reasons(reason_dir, plt, ymax)

        plt.legend()
        plt.tight_layout()
        # plt.show()
        if plot_file_path == None:
            plot_file_path = os.path.join(self.result_dir, 'hh_performance_line.png')
        else: 
            plot_file_path = plot_file_path
        plt.savefig(plot_file_path, dpi=150)
        print(f"plot_line_best: Plot saved as {plot_file_path}")

    def plot(self):
        self.plot_hh_multiplot()
        self.plot_hh_boxplot()
        self.plot_hh_boxplot_all()


if __name__ == "__main__":
    ps = visuSteps("/home/herget/UvA-git/hh_local/ASML-Faezeh_experiment_asml_firefly_dynamic_1732732391_20_iterations_30_steps_1732732391", "/home/herget/UvA-git/hh_local/ASML-Faezeh_experiment_asml_firefly_dynamic_1732732391_20_iterations_30_steps_1732732391")

    ps.plot()
