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


class visuSimulations:
    def __init__(self, simulationResultsDirPath):
        self.df = None
        self.rawdf = None
        self.simulationResultsDirPath = simulationResultsDirPath
        self._read_file_path = None
    
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
                'index_dupl': 'first'
            }
            for ax in axis_names:
                aggregation_functions[ax] =  'first'
            rawdf = rawdf.groupby(rawdf[fitness_col]).aggregate(aggregation_functions).reset_index()
        
        if max_perc_design_points != 1.0 or max_number_design_points != inf:
            max_perc_design_points = min(max_perc_design_points, 1.)
            max_perc_design_points = max(max_perc_design_points, 0.)
            max_perc_design_points_calc = math.ceil(len(rawdf.index) * max_perc_design_points)
            rawdf = rawdf.sort_values(by=fitness_col).iloc[:min(len(rawdf.index),max_number_design_points, max_perc_design_points_calc)]
            text = f"{min(len(rawdf.index),max_number_design_points, max_perc_design_points_calc)}/{alldp} DPs"
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

        localDF = self.rawdf
             
        plt.figure(figsize=figsize)
        plt.rc('text', usetex=True)
        title_tex = '{\\fontsize{20pt}{3em}\\selectfont{}'+title+'}\n{\\fontsize{12pt}{3em}\\selectfont{}'+str(len(localDF.index))+' Design Points}'
        if info is not None:
            title_tex += ' \n{\\fontsize{8pt}{3em}\\selectfont{}'+str(info)+'}'
        plt.title(r''+title_tex)

        bins = math.ceil(100 *( round(localDF[x_axis_name].max(), 2) - round(localDF[x_axis_name].min(), 2)))
        plt.hist(localDF[x_axis_name], bins=bins, kde=True)
        print(bins)

        if regression_Line:
            model = make_pipeline(PolynomialFeatures(2), LinearRegression())
            model.fit(np.array(localDF[x_axis_name]).reshape(-1, 1), localDF[y_axis_name])
            x_reg = np.arange(localDF[x_axis_name].min(), localDF[x_axis_name].max())
            y_reg = model.predict(x_reg.reshape(-1, 1))

            plt.plot(
                x_reg, 
                y_reg, 
                c='#ff7f0e',
                alpha=0.5,
                zorder=10,
            )

        print(self._figureOutLim(xlim_min, xlim_max, x_axis_name, localDF))
        if self._figureOutLim(xlim_min, xlim_max, x_axis_name, localDF)[0]:
            plt.xlim(self._figureOutLim(xlim_min, xlim_max, x_axis_name, localDF)[1])
        if self._figureOutLim(ylim_min, ylim_max, y_axis_name, localDF)[0]:
            plt.ylim(self._figureOutLim(ylim_min, ylim_max, y_axis_name, localDF)[1])
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.grid(True, alpha=0.5, ls="--", zorder=0)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)

    def plotFitnessHistogram(self, 
             filepath,
             title="Fitness Distribution", 
             x_axis_name=None,
             figsize=(6, 4),
             info = None, bins_multiplicator =1
        ):
        if self.axis_names and x_axis_name is None:
            x_axis_name = self.axis_names[0]
        if not self.axis_names and x_axis_name is None:
            raise ValueError("Global axis_names or x_axis_name/y_axis_name has not been set.")
                     
        localDF = self.rawdf
        plt.figure(figsize=figsize)
        plt.rc('text', usetex=True)
        title_tex = '{\\fontsize{20pt}{3em}\\selectfont{}'+title+'}\n{\\fontsize{12pt}{3em}\\selectfont{}'+str(len(localDF.index))+' Design Points}'
        if info is not None:
            title_tex += ' \n{\\fontsize{8pt}{3em}\\selectfont{}'+str(info)+'}'

        bins = math.ceil( \
            bins_multiplicator * 100 * ( \
                round(localDF[x_axis_name].max(), 2) \
                - round(localDF[x_axis_name].min(), 2 \
            )) \
        )
        ax = sns.displot(localDF[x_axis_name], bins=bins, kde=True)
        ax.set(title= r''+title_tex)
        ax.tight_layout()
        ax.savefig(filepath, dpi=150)
        
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


if __name__ == "__main__":
    ps = visuSimulations("")

    info = ps.extractData(
        "/home/herget/UvA-git/hh_local/design_point_metrics_ASML_20241127_113720.csv", 
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
        "/home/herget/UvA-git/hh_local/design_point_metrics_ASML_20241127_113720.csv", 
        ["wfpm", "cost"],
        # max_number_design_points = 100,
        max_perc_design_points = 0.5,
        min_fitness_value = 0.3,
        unifiyFitnessValues = False,
        index_col="SimulationID"
    )
    ps.plotFitnessHistogram(
        "/home/herget/UvA-git/hh_local/fitness_histogram_2.png", 
        info = info,
        x_axis_name = 'Fitness',
        bins_multiplicator=10
    )
