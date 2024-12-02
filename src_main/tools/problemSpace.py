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

class ProblemSpace:
    def __init__(self, coordinator_config_file_path, log_path):
        self.logger = logger("ProblemSpace", log_path, disabled=False)
        self.conf = Config(coordinator_config_file_path, log_path, "coordinator_config_manager")
        self._problems = {}    
    
    def create_problems(self, problems_space_name, num_replicas, create_problem_instance_func, *args, **kwargs):
        assert callable(create_problem_instance_func)

        probs = {}
        num_replicas = num_replicas if num_replicas > 0 else 1 

        self.logger.info(f"Creating {num_replicas} problems for {problems_space_name}.")
        for rep in range(num_replicas):
            probs[rep] = create_problem_instance_func(*args, **kwargs)
            probs[rep]['set_file_name_fitness_values']("fitness_values_"+str(problems_space_name)+"_replica_"+str(rep)+".json")        
            probs[rep]['set_space_name'](str(problems_space_name))

        self._problems[problems_space_name] = probs
        return probs

    def create_and_append_problem(self, problems_space_name, create_problem_instance_func, *args, **kwargs):
        assert callable(create_problem_instance_func)

        if hasattr(self._problems, problems_space_name):
            probs = self._problems[problems_space_name]
            replica_id = max(self._problems[problems_space_name].keys()) + 1
        else:
            probs = {}
            replica_id = 0

        self.logger.info(f"Creating 1 problem for {problems_space_name} and appending it.")
        probs[replica_id] = create_problem_instance_func(*args, **kwargs)
        probs[replica_id]['set_file_name_fitness_values']("fitness_values_"+str(problems_space_name)+"_replica_"+str(replica_id)+".json")        
        probs[replica_id]['set_space_name'](str(problems_space_name))

        self._problems[problems_space_name] = probs
        return probs

    def remove_problem(self, problems_space_name, replica_id=None):
        if hasattr(self._problems, problems_space_name):
            if replica_id is None:
                replica_id = max(self._problems[problems_space_name].keys())
            if hasattr(self._problems[problems_space_name], replica_id):
                self.logger.info(f"Removing problem {replica_id} of {problems_space_name}.")
                del self._problems[problems_space_name][replica_id]


    def has_problem_space(self, problems_space_name):
        if hasattr(self._problems, problems_space_name):
            return len(self._problems[problems_space_name]) > 0
        else:
            return  False

    def get_problems(self, problems_space_name):
        if self.has_problem_space(problems_space_name):
            return self._problems[problems_space_name]
        else:
            self.logger.warn(f"get_problems: The problem space {problems_space_name} does not exist or does not have any problems associated.")
            return []
