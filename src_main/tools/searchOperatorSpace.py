from src_main.tools.logger import logger, setLevelLogger
from src_main.tools.config_reader import Config
from src_main.tools.problemSpace import ProblemSpace

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

class SearchOperatorSpace:
    def __init__(self, coordinator_config_file_path, log_path):
        log_path = os.path.join(log_path, "search_operators")
        self.logger = logger("SearchOperatorSpace", log_path, disabled=False)
        self.conf = Config(coordinator_config_file_path, log_path, "searchOperatorSpace_config")
        self._problems = {}    
        self.logger.info("Setting up SearchOperatorSpace.")

        self.coordinator_config_file_path = coordinator_config_file_path
        self.log_path = log_path

        self.problem_spaces = {}
    
    def create_problems(self, problem_space_name, num_replicas, create_problem_instance_func, *args, **kwargs):
        assert callable(create_problem_instance_func)
        self.create_problem_space(problem_space_name)
        return self.problem_spaces[problem_space_name].create_problems(
            num_replicas, 
            create_problem_instance_func, 
            *args, **kwargs)

    def create_problem_space(self, problem_space_name, log_already_exist = False):
        if self.has_problem_space(problem_space_name):
            if log_already_exist:
                self.logger.info(f"Problem space {problem_space_name} already exist.")
        else:
            self.logger.info(f"Creating problem space {problem_space_name}.")
            self.problem_spaces[problem_space_name] = ProblemSpace(
                problem_space_name, 
                self.coordinator_config_file_path, 
                self.log_path
            )

    def remove_problem_space(self, problem_space_name):
        if self.has_problem_space(problem_space_name):
            self.logger.info(f"Removing problem space {problem_space_name}.")
            del self._problems[problem_space_name]

    def has_problem_space(self, problem_space_name):
        return problem_space_name in self.problem_spaces.keys()
    

    def has_problems(self, problem_space_name):
        if self.has_problem_space(problem_space_name):
            return self.problem_spaces[problem_space_name].has_problems()
        else:
            return False

    def get_problems(self, problem_space_name):
        if self.has_problem_space(problem_space_name):
            if self.problem_spaces[problem_space_name].has_problems():
                return self.problem_spaces[problem_space_name].get_problems()
            else:
                self.logger.warn(f"get_problems: The problem space {problems_space_name} does not have any problems associated.")
                return {}
        else:
            self.logger.warn(f"get_problems: The problem space {problems_space_name} does not exist .")
            return {}

    def get_all_problems_file_name_fitness_values(self):
        problems_file_name_fitness_values = {}
        for problem_space in self.problem_spaces.keys():
            for problem in self.get_problems(problem_space):
                problems_file_name_fitness_values[problem.get_file_name_fitness_values()] = problem
        return problems_file_name_fitness_values