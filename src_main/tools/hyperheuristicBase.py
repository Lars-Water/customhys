from src_main.tools.logger import logger, setLevelLogger
import src_main.tools.component_config as component_config
import src_main.tools.file_operations as fo
from src_main.tools.config_reader import Config
from src_main.tools.problemSpace import ProblemSpace
from src_main.tools.searchOperatorSpace import SearchOperatorSpace
from src_main.data import collect_data_INET

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
import threading
import copy
import math
# from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TextColumn, BarColumn
import rich.progress as prog
from tqdm import tqdm

from experiments import create_sim_custom_dummy, create_sim_inet_lans_dummy, create_sim_inet_lans_dummy_parallel
from src.manager import Manager
from src.utils.config_creator import WorkflowConfig
from src_main.external.customhys.customhys import hyperheuristic as hh

import uuid


class HyperHeuristicBase:
    search_operator_spaces = None

    def __init__(self, heur_coordinator, base_path, coordinator_config_file_path, experiment_config,  template_file_path, log_path=None, file_label_base="EXP-", experiment_name_base="experiment_", run_name = "experiment_"):       
        self.coordinator_config_file_path = coordinator_config_file_path
        self.log_path = log_path
        if self.log_path is None:
            self.log_path = os.path.join(self._base_path, "data/logs/")
        coordinator_log_path = Path(os.path.join(self.log_path, "coordinator"))

        self.conf = Config(coordinator_config_file_path, outputfolderpath=coordinator_log_path, name="HyperHeuristicBase_config")
        self.logger = logger("HyperHeuristicBase", coordinator_log_path, rich_handler=True, disabled=False)

        # Set logger level.
        logger_lvl = self.conf.tryGet("logger_lvl")
        setLevelLogger(self.logger, logger_lvl)

        self.search_operator_spaces = SearchOperatorSpace(self.coordinator_config_file_path, self.log_path)

        self.lock_hypers = threading.Lock()
        self.hypers = {}
        self._tasks_time_so_far = {}
        self.first_hyper_deactivated = False
        self.threads = {}
        self.procs = []
        self.save_runs = []

        self.heur_coordinator = heur_coordinator
        self.base_path = base_path
        self.experiment_config = experiment_config
        self.timestamp = self.heur_coordinator.timestamp
        self.timestamp_int = self.heur_coordinator.timestamp_int
        self.file_label_base = file_label_base
        self.experiment_name_base = experiment_name_base
        self.run_name = run_name
        self.template_file_path = template_file_path


        self.results_path = os.path.join(os.getcwd(), "data/raw/results/")
        if self.conf.tryGet("results_path") and self.conf.tryGet("results_path") is not None:
            self.results_path = self.conf.tryGet("results_path")

         # Configure Hyperheurstic Object.
        self.search_operators_spaces = self.experiment_config.tryGet('search_operators', 'spaces')
        self.minimum_amount_of_hhs = self.experiment_config.tryGet("search_operators", "minimum_amount_of_hhs")
        self.sync_steps_of_hhs = self.experiment_config.tryGet('search_operators', 'sync_steps_of_hhs')
        self.evaluate_after_steps = self.experiment_config.tryGet("search_operators", "evaluate_after_steps")
        self.pass_finalised_positions = self.experiment_config.tryGet('pass_finalised_positions')


        self.hh_parameters = self.experiment_config.tryGet('hh_parameters')
        self.nr_of_iterations = self.experiment_config.tryGet('hh_parameters', 'num_iterations')
        self.nr_of_agents = self.experiment_config.tryGet('hh_parameters', 'nr_of_agents')
        self.nr_of_steps = self.experiment_config.tryGet('hh_parameters', 'num_steps')
        self.num_replicas = self.experiment_config.tryGet('hh_parameters', 'num_replicas') if self.experiment_config.tryGet('hh_parameters', 'num_replicas') > 0 else 1 
        
        self.nr_of_design_queues = len(self.search_operators_spaces.keys()) * self.num_replicas
        self.coordinator_params = (self.base_path, self.coordinator_config_file_path, self.nr_of_agents, self.run_name)

        self.simulation_model_template_path = self.conf.tryGet("simulation_model", "simulation_model_paths", "simulation_model_template_path")
        
        self.agents_fitness_values_path = self.conf.tryGet("output_paths", "agents_fitness_values_path")

        self.barrier = threading.Barrier(len(self.search_operators_spaces.keys()))
        
    def run_multi_threaded(self):
        # with Progress() as progress:
        with prog.Progress(
                prog.SpinnerColumn(),
                prog.TextColumn("[progress.description]{task.description}", justify="right"),
                prog.TextColumn("[progress.percentage]{task.completed}/{task.total}", justify="right"),
                prog.BarColumn(),
                prog.TimeElapsedColumn(),
            ) as progress:
            self.progress = progress
            all_bars = []

            for search_operator_space_name in self.search_operators_spaces.keys():
                space = self.search_operators_spaces[search_operator_space_name]
                search_operator_space_path = space["path"]
                bars = [
                    self.progress.add_task(
                        f"[bold]{search_operator_space_name}[/bold] Step", 
                        total=self.nr_of_steps,
                        start=False,
                        completed=-1
                    ),
                    self.progress.add_task(
                        f"Iter", total=self.nr_of_iterations*self.num_replicas,
                        start=False
                    )
                ]
                all_bars += bars
                self.threads[search_operator_space_name] = threading.Thread(
                    target=self.hh_thread, 
                    args=(
                        search_operator_space_name,
                        search_operator_space_path,
                        bars
                    )
                )       
                
            for bar in all_bars:
                self.progress.start_task(bar)
            
            self.logger.info(f"Start {len(self.threads)} threads.")

            for search_operator_space_name, proc in self.threads.items():
                proc.start()
                self.procs.append(proc)

            for p in self.procs:
                p.join()

    def hh_thread(self, search_operator_space_name, search_operator_space_path, bars):
        # print("##### ("+search_operator_space_name+") Starting thread")
        self.logger.info(f'Starting thread for {search_operator_space_name}.')
        self.logger.debug(f' search_operator_space_name: {search_operator_space_name}\nself.num_replicas: {self.num_replicas}\nself.heur_coordinator.problemInstanceFunc(): {self.heur_coordinator.problemInstanceFunc()}\nself.template_file_path: {self.template_file_path}\n*self.heur_coordinator.get_boundaries(): {self.heur_coordinator.get_boundaries()}\nself.heur_coordinator.simulation_run: {self.heur_coordinator.simulation_run}\nself.agents_fitness_values_path: {self.agents_fitness_values_path}')
        self.search_operator_spaces.create_problems(
            search_operator_space_name, 
            self.num_replicas,  
            self.heur_coordinator.problemInstanceFunc(), 
            self.template_file_path, 
            *self.heur_coordinator.get_boundaries(), 
            self.heur_coordinator.simulation_run,
            self.agents_fitness_values_path
        )

        heuristic_space = component_config.determine_heuristic_space(search_operator_space_path)
        experiment_name = f"{self.experiment_name_base}_{search_operator_space_name}_{str(self.timestamp)}"
        file_label = f"{self.file_label_base}_{search_operator_space_name}_{self.nr_of_iterations}_iterations_{self.nr_of_steps}_steps_{str(self.timestamp)}"

        
        bar_steps = bars[0]
        bar_iter = bars[1]
        self.hypers[search_operator_space_name] = {
            "hh": hh.Hyperheuristic(
                heuristic_space=heuristic_space,
                # problems=probs,
                heur_coordinator=self.heur_coordinator,
                search_operator_space_name=search_operator_space_name,
                parameters=self.hh_parameters,
                file_label=file_label,
                pass_finalised_positions=self.pass_finalised_positions,
                updateMHProgress={ 
                    "advance": lambda x: self.progress.update(bar_iter, advance=x),
                    "start": lambda: self.progress.reset(bar_iter),
                },
                file_details= {
                    "experiment_name": experiment_name,
                    "hh_parameters": self.hh_parameters,
                    "timestamp": self.timestamp,
                    "timestamp_int": self.timestamp_int,
                    "search_operator_space_name": search_operator_space_name,
                    "template_file_path": self.template_file_path,
                    "coordinator_config": self.conf.conf()
                }
            ),
            "enabled": True,
            "steps": {},
            "best": {
                "step": 0
            },
            "progress_bar": {
                "advance": lambda x: self.progress.update(bar_steps, advance=x),
                "set_completed": lambda x: self.progress.update(bar_steps, completed=x),
                "finish": lambda x: self.progress.update(bar_steps, total=x, completed=x),
                "pause": lambda: self.progress.stop_task(bar_steps),
                "start": lambda: self.progress.start_task(bar_steps),
                "remove_iter": lambda: self.progress.remove_task(bar_iter)
            }
        }

        # self.progress.start_task(bar_steps)
        # time.sleep(2)

        # self._task_pause(bar_steps)
        # time.sleep(4)
        # self._task_resume(bar_steps)
        


        # Start timer for the heuristic run.
        start_time = time.time()

        # Start hyper-heuristic run.
        best_sol, best_perf, hist_curr, hist_best = self.hypers[search_operator_space_name]["hh"].solve()

        # End timer for the heuristic run.
        end_time = time.time()
        
        hh_run_meta_data = collect_data_INET.calculate_distinct_simulation_components(start_time, end_time, self.heur_coordinator)

        # Save the heuristic run data.
        # save_run_path = os.path.join(self.results_path, experiment_name)

        self.logger.info(f"Results for {search_operator_space_name}\nBest solution: {str(best_sol)}\nBest performance: {str(best_perf)}\nBest history: {str(hist_best)}\nCurrent history: {str(hist_curr)} \n {hh_run_meta_data}")

        self.save_runs.append({
            "search_operator_space_name": search_operator_space_name,
            "experiment_name": experiment_name,
            "best_solution": best_sol,
            "best_performance": best_perf,
            "current_history": hist_curr,
            "best_history": hist_best,
            "run_meta_data": hh_run_meta_data
        })
        
    def _task_pause(self, taskid):
        self._tasks_time_so_far[taskid] = self.progress._tasks[taskid].elapsed
        self.logger.warn(f"PAUSE {taskid}: {self._tasks_time_so_far[taskid] }")
        self.progress.stop_task(taskid)

    def _task_resume(self, taskid):
        time_so_far = self._tasks_time_so_far[taskid]
        # del self._tasks_time_so_far[taskid]
        completed = self.progress._tasks[taskid].completed +10
        self.progress._tasks[taskid].elapsed
        self.progress.reset(taskid, completed=completed)
        self.logger.warn(f"RESUME {taskid}: {time_so_far}, {completed}")
        time.sleep(5)
        # self.progress.update(taskid, elapsed=time_so_far)
        self.logger.warn(f"+++ RESUME {taskid}: {self.progress._tasks[taskid].elapsed}, {time_so_far}")
        self.progress.refresh()
        # self.progress.start_task(taskid)
        time.sleep(15)
        self.logger.warn(f"### RESUME {taskid}: {self.progress._tasks[taskid].elapsed}, {completed}")

    # All functions starting with "hh" are use by the hyperheuristic.py from customhys
    def hh_checkFinalization(self, 
                             search_operator_space_name,  
                             step, 
                             stag_counter,
                             best_performance,
                             current_performance):
        finalize = False
        reas = []
        # First order of business: Update our own data for comparisions
        # Lock: see below - tldr: avoid edge case
        with self.lock_hypers:
            self.hypers[search_operator_space_name]["progress_bar"]["set_completed"](int(step))
            self.hypers[search_operator_space_name]["best"] = {
                "step": step,
                "performance": best_performance
            }
            self.hypers[search_operator_space_name]["steps"][step] = {
                "performance": current_performance,
                "best": best_performance
            }
        # Do not check if we are already at the minimum amount of HHs
        # and only if we are the correct step interval
        self.logger.debug(f"hh_checkFinalization for {search_operator_space_name} (Step: {step}, stag_counter: {stag_counter}):\n{self.hypers[search_operator_space_name]}") # DEBUG
        enabled_hypers = self._get_num_enabled_hyper()
        if int(enabled_hypers) > int(self.minimum_amount_of_hhs) and self._is_evaluation_time(step):
            # If the check should always use the same step (sync_steps_of_hhs) then we need all HHs to reach this point, otherwise we check with the best we can 
            if self.sync_steps_of_hhs:
                self.logger.info(f"{search_operator_space_name} is waiting to check the extra finalization critera (synced). (Step: {step}, stag_counter: {stag_counter})")
                # self.hypers[search_operator_space_name]["progress_bar"]["pause"]()
                self.barrier.wait()
                # self.hypers[search_operator_space_name]["progress_bar"]["start"]()
                finalize, reas = self._checkFinalization_internal(search_operator_space_name, step, stag_counter, best_performance, current_performance)
            else: 
                # Lock is used, that only one HH at a time can be deactivated
                # edge case: one HH upates its best value while another one is deactivating a HH
                with self.lock_hypers:
                    finalize, reas = self._checkFinalization_internal(search_operator_space_name, step, stag_counter, best_performance, current_performance)
            self.logger.info(f"The extra finalize check for {search_operator_space_name} returns: {finalize}.") # DEBUG
        else:
            self.logger.debug(f"{search_operator_space_name} - Step: {step} - No extra check done")
        
        return finalize, reas

    def _checkFinalization_internal(self, 
                             search_operator_space_name,  
                             step, 
                             stag_counter,
                             best_performance,
                             current_performance):
        reas = []
        finalize = not self.hypers[search_operator_space_name]["enabled"]
        if finalize:
            reas.append("HH1")
        if self.experiment_config is not None:  
            if self._is_evaluation_time(step):
                finalize_is_worst = self._is_worst_hyper(search_operator_space_name, step)
                if finalize_is_worst:
                    reas.append("HH2")
                finalize =  finalize or finalize_is_worst
                            
        return finalize, reas
    
    def _is_evaluation_time(self, step):
        return self.evaluate_after_steps <= step and (step % self.evaluate_after_steps) == 0

    def _best_hyper(self):
        best_perf = 1
        best_name = None
        for name, hyper in self.hypers.items():
            if hyper["enabled"]:
                if hyper["best"]["performance"] < best_perf:
                    best_perf = hyper["best"]["performance"]
                    best_name = name

        return best_name

    def _is_worst_hyper(self, search_operator_space_name, step):
        worst_perf = -1
        worst_name = None
        for name, hyper in self.hypers.items():
            if hyper["enabled"]:
                # If any hh has not done enough steps yet, let all keep running
                if hyper["best"]["step"] < self.evaluate_after_steps:
                    return False
                # If the hyper has already done the specific step then us that step for comparision. Only really necessary if the evaluation is not synced
                elif step in hyper["steps"]:
                    if hyper["steps"][step]["best"] > worst_perf:
                        self.logger.debug(f'Found a new worst in step {step}: worst_perf: {worst_perf}, perf: {hyper["best"]["performance"]}, name: {name}')
                        worst_perf = hyper["best"]["performance"]
                        worst_name = name
                # Backup: Use overall best from the hyper
                else:
                    if hyper["best"]["performance"] > worst_perf:
                        self.logger.debug(f'Found a new worst: worst_perf: {worst_perf}, perf: {hyper["best"]["performance"]}, name: {name}')
                        worst_perf = hyper["best"]["performance"]
                        worst_name = name
        # If its the worst performing: disable
        if search_operator_space_name == worst_name:
            self.first_hyper_deactivated = True
            self.logger.warn(self.hypers)
            return True
        # Check whether one has been deactivate so far: If not, deactive one.
        # This needs to be done, so that after all HH achieved the minmum steps, one ist deactivated before the next iteration is done (only necessary if we do not sync the steps)
        # TODO: Check whether this actually works
        elif    not self.first_hyper_deactivated and \
                worst_name is not None and \
                not self.sync_steps_of_hhs:
            self.logger.info(f"First hyper will be deactivated at step {step}: {worst_name}")
            self.hypers[worst_name]["enabled"] = False
            self.first_hyper_deactivated = True
        return False

    def hh_disable_hh_and_distribute_Resources(self, search_operator_space_name, step, finalize_reason = []):
        with self.lock_hypers:
            self.hypers[search_operator_space_name]["enabled"] = False
            self.hypers[search_operator_space_name]["stopped"] = time.time()
            self.hypers[search_operator_space_name]["progress_bar"]["finish"](step)
            self.hypers[search_operator_space_name]["progress_bar"]["remove_iter"]()

            # Adjust barrier:
            enabled_hypers = self._get_num_enabled_hyper()
            self.barrier = threading.Barrier(enabled_hypers)

            if self.experiment_config is not None:
                optimize_utilization = self.experiment_config.tryGet("search_operators", "optimize_utilization")
                if optimize_utilization:
                    # Split available agents from deactivated HH to the rest
                    avail_agents = self.hypers[search_operator_space_name]["hh"].get_num_agents()
                    avail_agents_per_hyper = math.floor(avail_agents/enabled_hypers)
                    # Sometimes this cannot be distribute fairly. So we give the extra ones to the so far best performing HH
                    extra_avail_agents = avail_agents - avail_agents_per_hyper * enabled_hypers
                    best_hyper = self._best_hyper()
                    best_hyper_text = ""
                    if extra_avail_agents > 0:
                        best_hyper_text = f" Since there are {extra_avail_agents} agents which cannot be fairly distributed, those will be given to the currently best performaning Hyper Heuristic {best_hyper}."

                    self.logger.info(f"distribute_Resources: Hyper Heuristic {search_operator_space_name} is finalized and its {avail_agents} available agents will be distributed to the remaining {enabled_hypers} Hyper Heuristics ({avail_agents_per_hyper} per HH).{best_hyper_text}\nReasons: {', '.join(finalize_reason)}\n")

                    # json_out is used to write a file for better documentation
                    json_out = {
                        "step": step,
                        "finalize_reason": finalize_reason,
                    }

                    for hyper_space_name, hyper in self.hypers.items():
                        num_agents_avail = None
                        if hyper["enabled"]:
                            give_agents = avail_agents_per_hyper
                            if hyper_space_name == best_hyper:
                                give_agents += extra_avail_agents
                            num_agents_avail = hyper["hh"].give_avail_agents_for_next_step(give_agents)
                        json_out[hyper_space_name] = {
                            "num_agents_currently": hyper["hh"].get_num_agents(),
                            "num_agents_avail": num_agents_avail,
                            "hyper": {}
                        }
                        # Do not use the HH object
                        for key, val in hyper.items():
                            if key not in ["hh", "progress_bar"]:
                                json_out[hyper_space_name]["hyper"][key] = val

                    
                    dir_path = os.path.join(self.results_path, f"hh_hyper_{str(self.timestamp)}")
                    os.makedirs(dir_path, exist_ok=True)

                    with open(os.path.join(dir_path, f"hh_hyper_{search_operator_space_name}.json") , "w") as fp:
                        json.dump(json_out, fp, indent=4)
        
    def _get_num_enabled_hyper(self):
        enabled_hypers = 0
        for search_operator_space_name, hyper in self.hypers.items():
            if hyper["enabled"]:
                enabled_hypers += 1
        return enabled_hypers
