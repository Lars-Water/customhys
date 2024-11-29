import json
from math import inf
import os
import glob
from matplotlib import axis
import numpy as np
import pandas as pd
from paretoset import paretoset
import math 

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from pathlib import Path
import matplotlib.pyplot as plt

from visualization_base import visuBase



class visuParetoSet(visuBase):
    def __init__(self):
        super().__init__()

    def setData(self, df, sense=["min", "min"], rawdf=None):
        self.df = df
        self.rawdf = rawdf
        mask = paretoset(self.df, sense=sense)
        self.df_paretoset = self.df[mask]


    def plot(self, 
             filepath,
             title="Pareto Front", 
             label_x=None, label_y=None, 
             label_solution="Solution", label_efficient_solution="Efficient Solution", 
             x_axis_name=None, y_axis_name=None, 
             figsize=(6, 4), regression_Line=False,
             csv_all_filepath = None,
             csv_efficient_filepath = None, copy_efficient_solution_path_dir = None,
             info = None
        ):

        plt, x_axis_name, y_axis_name = self._plt(title, label_x, label_y, x_axis_name, y_axis_name, figsize, info)
 
       
        # plt.title(f"\n{len(self.df.index)} DP",  fontsize='small')

        plt.scatter(
            self.df[x_axis_name], 
            self.df[y_axis_name], 
            zorder=1, 
            label=label_solution,
            s=50, alpha=1, 
            c='#4c92c3'
        )

        plt.scatter(
            self.df_paretoset[x_axis_name],
            self.df_paretoset[y_axis_name],
            zorder=2,
            label=label_efficient_solution,
            s=100,
            alpha=1,
            edgecolors='#ff7f0e',
            linewidths=2,
            c='#4c92c3'
        )

        if regression_Line:
            model = make_pipeline(PolynomialFeatures(2), LinearRegression())
            model.fit(np.array(self.df_paretoset[x_axis_name]).reshape(-1, 1), self.df_paretoset[y_axis_name])
            x_reg = np.arange(self.df_paretoset[x_axis_name].min(), self.df_paretoset[x_axis_name].max())
            y_reg = model.predict(x_reg.reshape(-1, 1))

            plt.plot(
                x_reg, 
                y_reg, 
                c='#ff7f0e',
                alpha=0.5,
                zorder=10,

            )


        plt.tight_layout()
        plt.legend()
        plt.savefig(filepath, dpi=150)

        path_efficient_solutions = os.path.join(os.path.dirname(filepath), "efficient_solutions")
        if copy_efficient_solution_path_dir is not None:
            self.copySimulationsFiles(
                list(self.df.index.values), 
                copy_efficient_solution_path_dir, 
                path_efficient_solutions
            )

        if csv_efficient_filepath is not None or copy_efficient_solution_path_dir is not None:
            path = csv_efficient_filepath 
            if csv_efficient_filepath is None:
                path = path_efficient_solutions
            self.exportDataToCSV(path, self.df_paretoset)

        if csv_all_filepath is not None:
            self.exportDataToCSV(csv_all_filepath, self.df)


if __name__ == "__main__":
    ps = visuParetoSet()

    info = ps.extractData(
        "/home/herget/UvA-git/hh_local/design_point_metrics_ASML_20241127_193140.csv", 
        ["wfpm", "cost"],
        # max_number_design_points = 100,
        # max_perc_design_points = 0.25,d
        #min_fitness_value = 0.2,
        unifiyFitnessValues = False,
        index_col="SimulationID"
    )

    ps.plot(
        "/home/herget/UvA-git/hh_local/pareto_front.png", 
        regression_Line=False,
        # csv_efficient_filepath="/home/herget/UvA-git/hh_local/efficient.csv"
        info = info,
        # copy_efficient_solution_path_dir="/home/herget/UvA-git/hh_local/experiments_asml/data/campaign_asml/n1_w2_s2/20241127_191153/results"
    )
