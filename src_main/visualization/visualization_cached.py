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
 def extractData(self, 
                file_path, 
                axis_names, 
        ):
        self.axis_names = axis_names

        with open(file_path, 'r', encoding='utf-8') as json_file:
            try:
                # Load JSON content
                data = json.load(json_file)

        info = []

        steps_cached = {}
        for search_operator_name, steps in data["metadata"].items():
            for step, iterations in steps.items():
                for iteration, dp_stats in iterations.items():
                    if step not in steps_cached:
                        steps_cached[step] = { "cached": 0 }
                    steps_cached[step]["cached"] += dp_stats["num_dp_cached"]
        df = pd.DataFrame(steps_cached)

        self.setData(df)
        ret_info = None
        if len(info) > 0:
            ret_info ="(" + ", ".join(info) + ")"
        return ret_info

    def setData(self, df, rawdf=None):
        self.df = df
        self.rawdf = rawdf    


    def plotCachedperStep(self, 
             filepath,
             title="Fitness Distribution", 
             x_axis_name=None,
             figsize=(6, 4),
             info = None, bins_multiplicator =1
        ):
        x_axis_name, y_axis_name = self._plt_axis_names(x_axis_name)
                     
        localDF = self.df
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


if __name__ == "__main__":
    ps = visuSimulations("")

    info = ps.extractData(
        "/home/herget/UvA-git/hh_local/SML_20241203_015958_stats.json", 
        ["step", "cachedDP"],
    )

    ps.plotCachedperStep(
        "/home/herget/UvA-git/hh_local/cached_histogram.png", 
        info = info,
        x_axis_name = 'Steps',
    )