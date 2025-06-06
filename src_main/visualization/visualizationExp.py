import json
from math import inf
import os
import shutil
from matplotlib import axis
import numpy as np
import pandas as pd
import math 
import seaborn as sns
import errno
import argparse


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from pathlib import Path
import matplotlib.pyplot as plt

from visualization_cached import visuCached
from visualization_simulations import visuSimulations
from visualization_pareto import visuParetoSet
from visualization_steps import visuSteps


class visuExp:
    def __init__(self, configFilePath):
        if os.path.exists(configFilePath) and os.path.isfile(configFilePath):
            f = open(configFilePath)
            self.config = json.load(f)
        else:
            raise FileNotFoundError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
                configFilePath
            )
        os.makedirs(self.config["result_path"], exist_ok=True)
        
    def plot(self):
        # self.plotCached()
        self.plotParetoSet()
        # self.plotSimulations()
        # self.plotSteps()
    
    def plotCached(self):
        ps = visuCached(self.config["reason_dir"])
        info = ps.extractDataStats(self.config["stats"])
        ps.plotCachedperStep(
            os.path.join(self.config["result_path"], "cached_line.png"), 
            info = info,
            x_axis_name = 'Steps',
            y_axis_name = 'Number of cached DPs',
        )
        info = ps.extractData(
            self.config["metrics"],
            ["wfpm", "cost"],
            index_col="SimulationID"
        )
        ps.plotFitnessHistogram(
            os.path.join(self.config["result_path"], "cached_histogram.png"), 
            info = info,
            x_axis_name = 'Fitness',
            # bins_multiplicator=10
        )

    def plotParetoSet(self):
        ps = visuParetoSet()

        info = ps.extractData(
            self.config["metrics"],
            ["wfpm", "cost"],
            # max_number_design_points = 100,
            max_perc_design_points = 0.25,
            min_fitness_value = 0.2,
            unifiyFitnessValues = False,
            index_col="SimulationID"
        )

        ps.plot(
            os.path.join(self.config["result_path"], "pareto_front_min.png"), 
            regression_Line=False,
            csv_efficient_filepath=os.path.join(self.config["result_path"], "efficient.csv"),
            info = info,
            # copy_efficient_solution_path_dir="/home/herget/UvA-git/hh_local/experiments_asml/data/campaign_asml/n1_w2_s2/20241127_191153/results"
        )

    def plotSimulations(self):
        ps = visuSimulations("")

        info = ps.extractData(
            self.config["metrics"],
            ["wfpm", "cost"],
            # max_number_design_points = 100,
            # max_perc_design_points = 0.5,
            # min_fitness_value = 0.3,
            unifiyFitnessValues = False,
            index_col="SimulationID"
        )

        ps.plotFitnessHistogram(
            os.path.join(self.config["result_path"], "fitness_histogram_1.png"), 
            info = info,
            x_axis_name = 'Fitness',
        )

        info = ps.extractData(
            self.config["metrics"],
            ["wfpm", "cost"],
            # max_number_design_points = 100,
            # max_perc_design_points = 0.5,
            min_fitness_value = 0.25,
            unifiyFitnessValues = False,
            index_col="SimulationID"
        )
        ps.plotFitnessHistogram(
            os.path.join(self.config["result_path"], "fitness_histogram_2.png"), 
            info = info,
            x_axis_name = 'Fitness',
            bins_multiplicator=10
        )

        info = ps.extractData(
            self.config["metrics"],
            ["wfpm", "cost"],
            # max_number_design_points = 100,
            # max_perc_design_points = 0.5,
            min_fitness_value = 0.2,
            unifiyFitnessValues = False,
            index_col="SimulationID"
        )
        ps.plotFitnessHistogram(
            os.path.join(self.config["result_path"], "fitness_histogram_3.png"), 
            info = info,
            x_axis_name = 'Fitness',
            bins_multiplicator=20
        )

    def plotSteps(self):

        subfolders = [ f.path for f in os.scandir(self.config["mhs"]) if f.is_dir() ]
        ps = visuSteps(subfolders[0], self.config["result_path"])
        ps.plot_line_best(self.config["reason_dir"])
        ps.plot_agents(self.config["reason_dir"])
        for folder in subfolders:
            ps.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the exp visu..")
    parser.add_argument('--config', required=True, help='Choose which experiment to visu')
    args = parser.parse_args()

    config = args.config
    ps = visuExp(config)

    ps.plot()
