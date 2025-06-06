import json
from math import inf
import os
import shutil
from matplotlib import axis
import numpy as np
import pandas as pd
import math 
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from pathlib import Path
import matplotlib.pyplot as plt


class visuBase:
    def __init__(self):
        self.df = None
        self.rawdf = None
        self._read_file_path = None
        self._read_reason_dir = None
    
    def _readData(self, file_path, axis_names):
        if not hasattr(self, '_read_rawdf') or self._read_file_path != file_path:
            rawdf = pd.read_csv(file_path)
            rawdf = rawdf.dropna(subset=axis_names)
            rawdf['amount'] = 1
            self._read_file_path = file_path
            self._read_rawdf = rawdf
        else:
            print("Data have been read before")
            rawdf = self._read_rawdf
        
        return rawdf

    def extractData(self, 
                         file_path, 
                         axis_names, 
                         index_col=None, fitness_col="Fitness", 
                         min_fitness_value=inf, 
                         max_number_design_points=inf, max_perc_design_points=1.0,
                         unifiyFitnessValues=False
        ):
        self.axis_names = axis_names
        self.index_col = index_col
        rawdf = self._readData(file_path, axis_names)
        alldp = len(rawdf.index)

        info = []

        if index_col is not None:
            rawdf['index_dupl'] = rawdf[index_col]

        if unifiyFitnessValues:
            aggregation_functions = {
                'amount': 'count',
                index_col: tuple,
                'index_dupl': 'first',
                'Cached': 'count',
            }
            for ax in axis_names:
                aggregation_functions[ax] =  'first'
            rawdf = rawdf.groupby(rawdf[fitness_col]).aggregate(aggregation_functions).reset_index()
        
        if max_perc_design_points != 1.0 or max_number_design_points != inf:
            max_perc_design_points = min(max_perc_design_points, 1.)
            max_perc_design_points = max(max_perc_design_points, 0.)
            max_perc_design_points_calc = math.ceil(len(rawdf.index) * max_perc_design_points)

            limit = min(len(rawdf.index),max_number_design_points, max_perc_design_points_calc)
            text = f"{limit}/{alldp} DPs"

            rawdf = rawdf.sort_values(by=fitness_col).iloc[:limit]
            
            if max_number_design_points < max_perc_design_points_calc:
                text += f" (Limit: {max_number_design_points})"
            else:
                text += f" (Limit: {max_perc_design_points*100}\%)"
            
            info.append(text)

        if min_fitness_value != inf:
            rawdf = rawdf[rawdf[fitness_col] <= min_fitness_value]
            if len(info) == 0:
                info.append(f"{len(rawdf.index)}/{alldp} DPs")
            info.append(f"Min Fitness: {min_fitness_value}")        
        
        df = pd.DataFrame({
            axis_names[0]: rawdf[axis_names[0]].to_numpy(),
            axis_names[1]: rawdf[axis_names[1]].to_numpy()
        })
        if index_col is not None:
            if unifiyFitnessValues:
                rawdf = rawdf.rename(columns={index_col: f'{index_col}_grouped', 'index_dupl': index_col})
            else:
                rawdf = rawdf.drop(columns=['index_dupl'])
            df[index_col] = rawdf[index_col].to_numpy()
            rawdf = rawdf.set_index(index_col)
            df = df.set_index(index_col)

        self.setData(df, rawdf=rawdf)
        ret_info = None
        if len(info) > 0:
            ret_info ="(" + ", ".join(info) + ")"
        return ret_info

    def setData(self, df, rawdf=None):
        self.df = df
        self.rawdf = rawdf   

    def readDynamicSelectionReasons(self, reason_dir):
        reasons = {}
        reasons_steps = {}
        max_step_so_far = 0

        if not hasattr(self, '_read_reason_dir') or self._read_reason_dir != reason_dir:
            for filename in os.listdir(reason_dir):
                # Construct full file path
                file_path = os.path.join(reason_dir, filename)
                if filename.endswith('.json') and os.path.isfile(file_path) and not filename.endswith('config.json') and not filename.endswith("solutions.json"):
                    with open(file_path, 'r', encoding='utf-8') as json_file:
                        data = json.load(json_file)
                        if max_step_so_far < data["step"]:
                            for key in data.keys():
                                if type(data[key]) == dict and key not in ["step", "finalize_reason", "search_operator_space_name"]:
                                    steps_int = [int(x) for x in data[key]["hyper"]["steps"].keys()]
                                    if data[key]["num_agents_avail"] is not None:
                                        num_agents_avail = int(data[key]["num_agents_avail"])
                                    else:
                                        num_agents_avail = 0
                                    reasons[key] = {
                                        "step": max(steps_int),
                                        "steps": data[key]["hyper"]["steps"],
                                        "best": data[key]["hyper"]["best"],
                                        "num_agents_currently": int(data[key]["num_agents_currently"]),
                                        "num_agents_avail": num_agents_avail,
                                        "num_agents_all": num_agents_avail + int(data[key]["num_agents_currently"])
                                    }
                                    if "stopped" in data[key]["hyper"]:
                                        reasons[key]["stopped"] = data[key]["hyper"]["stopped"]
                            max_step_so_far = data["step"]
                        if "finalize_reason" in data.keys():
                            reasons_steps[data["step"]] = data["finalize_reason"]

            for key, reason in reasons.items():
                if reason["step"] in reasons_steps.keys():
                    reason["finalize_reason"] = reasons_steps[reason["step"]]
                else:
                    reason["finalize_reason"] = []

            self._read_reason_dir = reason_dir
            self.reasons = reasons
        return self.reasons
    
    def readNumAgens(self, reason_dir):
        agents = {}

        for filename in os.listdir(reason_dir):
            # Construct full file path
            file_path = os.path.join(reason_dir, filename)
            if filename.endswith('.json') and os.path.isfile(file_path) and not filename.endswith('config.json') and not filename.endswith("solutions.json"):
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    step = data["step"]
                    agents[key] = {
                        "x": [],
                        "y": []
                    }
                    for key in data.keys():
                        if type(data[key]) == dict and key not in ["step", "finalize_reason", "search_operator_space_name"]:
                            agents[key] = {

                            }


        return agents

    def plot_Reasons(self, reason_dir, plt, ymax):
        if reason_dir:
            reasons = self.readDynamicSelectionReasons(reason_dir)
            done_x = []
            for search_operator, reason in reasons.items():
                plt.vlines(x=[reason["step"]], ymin=0, ymax=ymax, color = 'gray', linestyles='dotted')
                if reason["step"] in done_x:
                    x = reason["step"] + 0.5
                else: 
                    x = reason["step"] 
                plt.text(x+ 0.1, ymax, f'{search_operator}: {"|".join(reason["finalize_reason"])}', rotation=90, verticalalignment='top')
                done_x.append(x)

    def exportDataToCSV(self, path, filter_df, backup_df = None):
        if backup_df is None:
            backup_df = filter_df
        if self.rawdf is not None:
            self.rawdf.filter(items=list(filter_df.index.values), axis=0) \
                        .to_csv(path, index=True, index_label=self.index_col)
        else:
            backup_df.to_csv(path, index=True)

    def copySimulationsFiles(self, SimulationIds, directory_path, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for sim_id in SimulationIds:
            sim_path = os.path.join(directory_path, sim_id)
            target_path = os.path.join(target_dir, sim_id)
            self.__copy_folder(sim_path, target_path)

    def __copy_folder(self, old, new):
        if (os.path.isdir(new)):
            shutil.rmtree(new)
        shutil.copytree(old, new)

    def _plt(self,
             title="Plot", 
             label_x=None, label_y=None, 
             x_axis_name=None, y_axis_name=None, 
             figsize=(6, 4), 
             info = None
        ):
        x_axis_name, y_axis_name = self._plt_axis_names(x_axis_name, y_axis_name)
        if label_x is None:
            label_x = str(x_axis_name)
        if label_y is None:
            label_y = str(y_axis_name)

        plt.figure(figsize=figsize)
        plt.rc('text', usetex=True)
        plt.title(r''+self._plt_title_tex(title, info))
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.grid(True, alpha=0.5, ls="--", zorder=0)

        return plt, x_axis_name, y_axis_name
    
    def _plt_axis_names(self, x_axis_name=None, y_axis_name=None):
        if hasattr(self, "axis_names") and x_axis_name is None:
            x_axis_name = self.axis_names[0]
        if hasattr(self, "axis_names") and y_axis_name is None:
            y_axis_name = self.axis_names[1]
        if not hasattr(self, "axis_names") and (y_axis_name is None or x_axis_name is None):
            raise ValueError("Global axis_names or x_axis_name/y_axis_name has not been set.")
        return x_axis_name, y_axis_name
    
    def _plt_title_tex(self, title, info, df = None):
        if df is None:
            df = self.df
        title_tex = '{\\fontsize{20pt}{3em}\\selectfont{}'+title+'}\n{\\fontsize{12pt}{3em}\\selectfont{}Best '+str(len(df.index))+' Design Points}'
        if info is not None:
            title_tex += ' \n{\\fontsize{8pt}{3em}\\selectfont{}'+str(info)+'}'
        return title_tex
        
    def _figureOutLim(self, lim_min, lim_max, axis_name, localDF):
        lim = [inf, inf]
        if lim_min is not None:
            lim[0] = lim_min
        if lim_max is not None:
            lim[1] = lim_max
        if lim_max is not None and lim_min is None:
            lim[0] = math.ceil(localDF[axis_name].min() * 0.9) 
        if lim_min is not None and lim_max is None:
            lim[1] = math.ceil(localDF[axis_name].max() * 1.2) 
        print(f'{axis_name}: {lim}')
        return ((lim[0] != inf), lim)
