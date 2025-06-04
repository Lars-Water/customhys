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

from visualization_base import visuBase

class visuSimulations(visuBase):
    def __init__(self, simulationResultsDirPath):
        super().__init__()
        self.simulationResultsDirPath = simulationResultsDirPath

    def setData(self, df, rawdf=None):
        self.df = df
        self.rawdf = rawdf    

    def plotFitnessDistribution(self, 
             filepath,
             title="Fitness Distribution", 
             label_x=None, label_y=None, 
             x_axis_name=None, y_axis_name=None, 
             figsize=(6, 4), regression_Line=False,
             info = None,
             xlim_min = None, ylim_min = None,
             xlim_max = None, ylim_max = None
        ):

        localDF = self.rawdf
        plt, x_axis_name, y_axis_name = self._plt(title, label_x, label_y, x_axis_name, y_axis_name, figsize, info)
 
        bins = math.ceil(100 *( round(localDF[x_axis_name].max(), 2) - round(localDF[x_axis_name].min(), 2)))
        plt.hist(localDF[x_axis_name], bins=bins, kde=regression_Line)

        if self._figureOutLim(xlim_min, xlim_max, x_axis_name, localDF)[0]:
            plt.xlim(self._figureOutLim(xlim_min, xlim_max, x_axis_name, localDF)[1])
        if self._figureOutLim(ylim_min, ylim_max, y_axis_name, localDF)[0]:
            plt.ylim(self._figureOutLim(ylim_min, ylim_max, y_axis_name, localDF)[1])

        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        print(f"plotFitnessDistribution: Plot saved as {filepath}")

    def plotFitnessHistogram(self, 
             filepath,
             title="Fitness Distribution", 
             x_axis_name=None,
             figsize=(6, 4),
             info = None, bins_multiplicator =1
        ):
        x_axis_name, y_axis_name = self._plt_axis_names(x_axis_name)
                     
        localDF = self.rawdf
        plt.figure(figsize=figsize)
        plt.rc('text', usetex=True)

        print(localDF[x_axis_name].max(), localDF[x_axis_name].min())

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
    ps = visuSimulations("")

    info = ps.extractData(
        "/home/herget/UvA-git/hh_local/design_point_metrics_ASML_20241127_193140.csv", 
        ["wfpm", "cost"],
        # max_number_design_points = 100,
        # max_perc_design_points = 0.5,
        # min_fitness_value = 0.3,
        unifiyFitnessValues = False,
        index_col="SimulationID"
    )

    ps.plotFitnessHistogram(
        "/home/herget/UvA-git/hh_local/fitness_histogram_1.png", 
        info = info,
        x_axis_name = 'Fitness',
    )

    info = ps.extractData(
        "/home/herget/UvA-git/hh_local/design_point_metrics_ASML_20241127_193140.csv", 
        ["wfpm", "cost"],
        # max_number_design_points = 100,
        max_perc_design_points = 0.5,
        # min_fitness_value = 0.,
        unifiyFitnessValues = False,
        index_col="SimulationID"
    )
    ps.plotFitnessHistogram(
        "/home/herget/UvA-git/hh_local/fitness_histogram_2.png", 
        info = info,
        x_axis_name = 'Fitness',
        bins_multiplicator=10
    )

    info = ps.extractData(
        "/home/herget/UvA-git/hh_local/design_point_metrics_ASML_20241127_193140.csv", 
        ["wfpm", "cost"],
        # max_number_design_points = 100,
        max_perc_design_points = 0.25,
        # min_fitness_value = 0.5,
        unifiyFitnessValues = False,
        index_col="SimulationID"
    )
    ps.plotFitnessHistogram(
        "/home/herget/UvA-git/hh_local/fitness_histogram_3.png", 
        info = info,
        x_axis_name = 'Fitness',
        bins_multiplicator=20
    )
