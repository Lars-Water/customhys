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


class visuParetoSet:
    def __init__(self):
        self.df = None
        self.rawdf = None
    

    def extractParetoset(self, 
                         file_path, 
                         axis_names, 
                         index_col=None, fitness_col="Fitness", 
                         min_fitness_value=inf, 
                         max_number_design_points=inf, max_perc_design_points=1.0,
                         unifiyFitnessValues=False
        ):
        self.axis_names = axis_names
        self.index_col = index_col
        rawdf = pd.read_csv(file_path)
        rawdf = rawdf.dropna(subset=axis_names)
        rawdf['amount'] = 1

        if index_col is not None:
            rawdf['index_dupl'] = rawdf[index_col]

        if unifiyFitnessValues:
            aggregation_functions = {
                'amount': 'count',
                index_col: tuple,
                'index_dupl': 'first'
            }
            for ax in axis_names:
                aggregation_functions[ax] =  'first'
            rawdf = rawdf.groupby(rawdf[fitness_col]).aggregate(aggregation_functions).reset_index()
        
        if min_fitness_value != inf:
            rawdf = rawdf[rawdf[fitness_col] <= min_fitness_value]

        if max_perc_design_points != 1.0 or max_number_design_points != inf:
            max_perc_design_points = min(max_perc_design_points, 1.)
            max_perc_design_points = max(max_perc_design_points, 0.)
            max_perc_design_points_calc = math.ceil(len(rawdf.index) * max_perc_design_points)
            rawdf = rawdf.sort_values(by=fitness_col).iloc[:min(len(rawdf.index),max_number_design_points, max_perc_design_points_calc)]
        
        
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

        self.setParetoset(df, rawdf=rawdf)

    def setParetoset(self, df, sense=["min", "min"], rawdf=None):
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
             csv_efficient_filepath = None
        ):
        if self.axis_names and x_axis_name is None:
            x_axis_name = self.axis_names[0]
        if self.axis_names and y_axis_name is None:
            y_axis_name = self.axis_names[1]
        if not self.axis_names and (y_axis_name is None or x_axis_name is None):
            raise ValueError("Global axis_names or x_axis_name/y_axis_name has not been set.")
        
        if label_x is None:
            label_x = str(x_axis_name)
        if label_y is None:
            label_y = str(y_axis_name)
             
        plt.figure(figsize=figsize)
        plt.rc('text', usetex=True)
        title_tex = '{\\fontsize{20pt}{3em}\\selectfont{}'+title+'}\n{\\fontsize{12pt}{3em}\\selectfont{}Best '+str(len(self.df.index))+' Design Points}'
        plt.title(r''+title_tex)
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
                zorder=0,

            )

        plt.legend()
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.grid(True, alpha=0.5, ls="--", zorder=0)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)

        if csv_all_filepath is not None:
            if self.rawdf is not None:
                self.rawdf.filter(items=list(self.df.index.values), axis=0) \
                          .to_csv(csv_all_filepath, index=True, index_label=self.index_col)
            else:
                self.df.to_csv(csv_all_filepath, index=True)

        if csv_efficient_filepath is not None:
            if self.rawdf is not None:
                self.rawdf.filter(items=list(self.df_paretoset.index.values), axis=0) \
                          .to_csv(csv_efficient_filepath, index=True, index_label=self.index_col)
            else:
                self.df_paretoset.to_csv(csv_efficient_filepath, index=True)

if __name__ == "__main__":
    ps = visuParetoSet("")

    ps.extractParetoset(
        "/home/herget/UvA-git/hh_local/design_point_metrics_ASML_20241126_133237.csv", 
        ["wfpm", "cost"],
        max_number_design_points = 100,
        max_perc_design_points = 0.5,
        min_fitness_value = 0.3,
        unifiyFitnessValues = False,
        index_col="SimulationID"
    )

    ps.plot(
        "/home/herget/UvA-git/hh_local/pareto_front.png", 
        csv_all_filepath="/home/herget/UvA-git/hh_local/all.csv",
        csv_efficient_filepath="/home/herget/UvA-git/hh_local/efficient.csv"
    )
