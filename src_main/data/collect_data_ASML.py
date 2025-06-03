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

from .collect_dataBase import DataCollectorBase

class DataCollectorASML(DataCollectorBase):

    def __init__(self, weight_wfpm, weight_cost, dir_design_points_metrics_output, run_name=None):
        """
        Initialize the CollectData object.

        Args:
            weight_latency (float): The weight for the latency metric.
            weight_cost (float): The weight for the cost metric.
            dir_design_points_metrics_output (str): The directory path for storing design points metrics output.

        Returns:
            None
        """
        if run_name is None:
            run_name = "default"
        super().__init__(dir_design_points_metrics_output, run_name=run_name)
        self.weight_wfpm = weight_wfpm
        self.weight_cost = weight_cost


    def store_design_point_metrics(self, wfpm, cost, sim_uid, heuristic_name):

        # Determine weighted metric values._
        adjusted_wfpm = wfpm * self.weight_wfpm
        adjusted_cost = cost * self.weight_cost

        # Append the adjusted values to the design points metrics storage.
        append_df = pd.DataFrame([[sim_uid, wfpm, cost, adjusted_wfpm, adjusted_cost]],
                                columns=['SimulationID', 'wfpm', 'cost', 'AdjustedWFPM', 'AdjustedCost'])
        
        self._store_design_point_metrics_df(heuristic_name, append_df)