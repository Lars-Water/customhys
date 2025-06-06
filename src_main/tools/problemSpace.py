from src_main.tools.logger import logger, setLevelLogger
from src_main.tools.config_reader import Config

import os
import glob
import shutil
import re
import pandas as pd
import numpy as np
import time
import json
import sys
from pathlib import Path
import glob
import itertools

from src_main.tools.local_parallelization.rpc import requires_main_process, callable_from_main

class ProblemSpace:
    @requires_main_process
    def __init__(self, problem_space_name, coordinator_config_file_path, log_path):
        self.logger = logger(f"problemSpace_{problem_space_name}", log_path, disabled=False)
        self.conf = Config(coordinator_config_file_path, log_path, f"problemSpace_{problem_space_name}_config")
        self.problem_space_name = problem_space_name
        
        self.logger.info(f"Setting up problem space {self.problem_space_name}.")
        self._problems = {}    
    
    @requires_main_process
    def create_problems(self, num_replicas, create_problem_instance_func, *args, **kwargs):
        assert callable(create_problem_instance_func)

        probs = {}
        num_replicas = num_replicas if num_replicas > 0 else 1 

        self.logger.info(f"Creating {num_replicas} problems.")
        for rep in range(num_replicas):
            self._problems[rep] = create_problem_instance_func(*args, **kwargs)
            self._problems[rep]['set_file_name_fitness_values']("fitness_values_"+str(self.problem_space_name)+"_replica_"+str(rep)+".json")        
            self._problems[rep]['set_space_name'](str(self.problem_space_name))

        return self._problems

    @requires_main_process
    def create_and_append_problem(self, create_problem_instance_func, *args, **kwargs):
        assert callable(create_problem_instance_func)

        if self.has_problems():
            replica_id = max(self._problems.keys()) + 1
        else:
            replica_id = 0

        self.logger.info(f"Creating 1 problem and appending it.")
        self._problems[replica_id] = create_problem_instance_func(*args, **kwargs)
        self._problems[replica_id]['set_file_name_fitness_values']("fitness_values_"+str(self.problem_space_name)+"_replica_"+str(replica_id)+".json")        
        self._problems[replica_id]['set_space_name'](str(self.problem_space_name))

        return self._problems

    @requires_main_process
    def remove_problem(self, replica_id=None):
        if self.has_problems():
            if replica_id is None:
                replica_id = max(self._problems.keys())
            if replica_id in self._problems.keys():
                self.logger.info(f"Removing problem {replica_id}.")
                del self._problems[replica_id]

    @requires_main_process
    def has_problems(self):
        return len(self._problems) > 0

    @requires_main_process
    def get_problems(self):
        if self.has_problems():
            return self._problems
        else:
            self.logger.warn(f"get_problems: The problem space does not have any problems associated.")
            return {}
