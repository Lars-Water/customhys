import json
from math import inf
import os
import glob
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
from datetime import timedelta
from visualization_base import visuBase

class visuCached(visuBase):
    def __init__(self, reason_dir=None):
        super().__init__()
        self.reason_dir = reason_dir
        
    def extractDataStats(self, file_path):

        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        
        exp_name = str(str(file_path).split("/")[-1]).split("_stats")[0]

        info = [f'Total: {data["general"]["num_dp"]}', f'Cached: {data["general"]["num_dp_cached"]}']
        if "environment_time" in data["general"]:
            td = timedelta(seconds=math.floor(data["general"]["environment_time"]))
            info.append(f'Runtime: {td}')
            
        info.append(f"Exp: {exp_name}")

        steps_cached = {}
        steps_cached_problems = {}
        for search_operator_name, steps in data["metadata"].items():
            if search_operator_name.startswith("problem_"):
                search_operator_name.split("_")[1:]
                if search_operator_name not in steps_cached_problems:
                    steps_cached_problems[search_operator_name] = { }
                for step, iterations in steps.items():
                    step = step.split("_")[1]
                    for iteration, dp_stats in iterations.items():
                        if step not in steps_cached:
                            steps_cached[step] = { "cached": 0 }
                        if step not in steps_cached_problems[search_operator_name]:
                            steps_cached_problems[search_operator_name][step] = 0 
                        steps_cached[step]["cached"] += dp_stats["num_dp_cached"]
                        steps_cached_problems[search_operator_name][step] += dp_stats["num_dp_cached"]
        df = pd.DataFrame(steps_cached).transpose()

        rawdf = pd.DataFrame(steps_cached_problems)

        self.setData(df, rawdf)
        ret_info = None
        if len(info) > 0:
            ret_info ="(" + ", ".join(info) + ")"
        return ret_info

    def setData(self, df, rawdf=None):
        self.df = df
        self.rawdf = rawdf    

    def _plt_title_tex(self, title, info, df = None):
        if df is None:
            df = self.df
        title_tex = '{\\fontsize{20pt}{3em}\\selectfont{}'+title+'}'
        if info is not None:
            title_tex += ' \n{\\fontsize{8pt}{3em}\\selectfont{}'+str(info)+'}'
        return title_tex
    
    def plotCachedperStep(self, 
             filepath,
             title="Cached DPs", 
             x_axis_name=None, y_axis_name=None,
             figsize=(6, 4),
             info = None,
             label_x=None, label_y=None, 
        ):
        x_axis_name, y_axis_name = self._plt_axis_names(x_axis_name, y_axis_name)
                     
        localDF = self.df
        plt, x_axis_name, y_axis_name = self._plt(title, label_x, label_y, x_axis_name, y_axis_name, figsize, info)

        plt.plot(localDF['cached'], label=f'Combined')
        for column_name, column in self.rawdf.items():
            plt.plot(column, label=column_name)


        if self.reason_dir:
            self.plot_Reasons(self.reason_dir, plt, max(localDF['cached']))

        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(filepath, dpi=150)
        print(f"plotCachedperStep: Plot saved as {filepath}")


    def plotFitnessHistogram(self, 
             filepath,
             title="Cached DPs Distribution", 
             x_axis_name=None,
             figsize=(6, 4),
             info = None, bins_multiplicator =1
        ):
        x_axis_name, y_axis_name = self._plt_axis_names(x_axis_name)

        localDF = self.rawdf.loc[self.rawdf.Cached]
        info = ", ".join([f'Total: {len(self.rawdf)}', f'Cached: {len(localDF)}'])

        plt.figure(figsize=figsize)
        plt.rc('text', usetex=True)

        bins = math.ceil( \
            bins_multiplicator * 100 * ( \
                round(localDF[x_axis_name].max(), 2) \
                - round(localDF[x_axis_name].min(), 2 \
            )) \
        )
        ax = sns.displot(localDF[x_axis_name], bins=bins, kde=True)
        ax.set(title= r''+self._plt_title_tex(title, info, localDF))
        ax.tight_layout()
        ax.savefig(filepath, dpi=150)
        print(f"plotFitnessHistogram: Plot saved as {filepath}")

if __name__ == "__main__":
    ps = visuCached("")

    info = ps.extractDataStats(
        "/home/herget/UvA-git/hh_local/hh_test_05_12_22_53/data/rawASML/ASML_20241205_201227_stats.json"
    )

    ps.plotCachedperStep(
        "/home/herget/UvA-git/hh_local/cached_line.png", 
        info = info,
        x_axis_name = 'Steps',
        y_axis_name = 'Number of cached DPs',
    )

    info = ps.extractData(
        "/home/herget/UvA-git/hh_local/hh_test_05_12_22_53/design_points/design_point_metrics_ASML_20241205_201227.csv",
        ["wfpm", "cost"],
        index_col="SimulationID"
    )

    ps.plotFitnessHistogram(
        "/home/herget/UvA-git/hh_local/cached_histogram.png", 
        info = info,
        x_axis_name = 'Fitness',
        # bins_multiplicator=10
    )