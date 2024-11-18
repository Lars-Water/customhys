import math
import numpy as np
import pandas as pd
import json
import os
import re
import glob
from datetime import datetime
import time
from pathlib import Path

from src_main.tools.config_reader import Config
from src_main.visualization import visualization
import src_main.tools.component_config as component_config
import src_main.data.sim_configurations as sim_configurations

from .collect_data import DataCollector

class DataCollectorASML(DataCollector):

    def __init__(self, weight_wfpm, dir_design_points_metrics_output):
        """
        Initialize the CollectData object.

        Args:
            weight_latency (float): The weight for the latency metric.
            weight_cost (float): The weight for the cost metric.
            dir_design_points_metrics_output (str): The directory path for storing design points metrics output.

        Returns:
            None
        """
        self.weight_wfpm = weight_wfpm
        self.dir_design_points_metrics_output = dir_design_points_metrics_output


    def store_design_point_metrics(self, wfpm, sim_uid, heuristic_name):
        """
        Stores the design point metrics in a CSV file.

        Args:
            latency_df (pandas.DataFrame): DataFrame containing latency values.
            cost_df (pandas.DataFrame): DataFrame containing cost values.
            sim_uid (str): Unique identifier for the simulation.

        Returns:
            None
        """
        # Define the file output path.
        os.makedirs(self.dir_design_points_metrics_output, exist_ok=True)
        append_design_points_metric_output_file = os.path.join(self.dir_design_points_metrics_output, f"design_point_metrics_{heuristic_name}.csv")

        # Determine weighted metric values._
        adjusted_wfpm = wfpm * self.weight_wfpm

        # Append the adjusted values to the design points metrics storage.
        append_df = pd.DataFrame([[sim_uid, adjusted_wfpm]],
                                columns=['SimulationID', 'AdjustedWFPM'])
        append_df.to_csv(append_design_points_metric_output_file, mode='a', header=not os.path.exists(append_design_points_metric_output_file), index=False)
