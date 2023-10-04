import json
import os
import shutil
import sys

from pathlib import Path, PosixPath

from src.utils.config_reader import Config
from src.utils.logger import logger
from src.utils.stats import Stats

class OutputHandler:
    logger = None
    cnf = None
    sims_path = None
    config_path = None
    stats = None

    def __init__(self, config_path, logs_path):
        self.logs_path = logs_path
        self.config_path = config_path
        os.makedirs(self.logs_path, exist_ok=True)

        self.stats = Stats()

        self.logger = logger("output_handler", Path(self.logs_path))

        self.logger.info("Reading in config file: {}".format(self.config_path))
        self.cnf = Config(Path(self.config_path), Path(self.logs_path), "config_output_handler")

        self.sims_path = self.cnf.tryGet("sims_path")
        self.logger.info("Sims path: {}".format(self.sims_path))

        self.global_results = self.cnf.tryGet("global_sim_results")
        os.makedirs(self.global_results, exist_ok=True)
        self.logger.info("Global results path: {}".format(self.global_results))

        self.global_logs = self.cnf.tryGet("global_sim_logs")
        os.makedirs(self.global_logs, exist_ok=True)
        self.logger.info("Global logs path: {}".format(self.global_logs))

        self.global_runtime = self.cnf.tryGet("global_sim_runtime")
        os.makedirs(self.global_runtime, exist_ok=True)
        self.logger.info("Global runtime path: {}".format(self.global_runtime))

    def sim_global_results_path(self, sim_instance):
        return os.path.join(self.global_results, str(sim_instance.uid))

    def sim_global_logs_path(self, sim_instance):
        return os.path.join(self.global_logs, str(sim_instance.uid))

    def sim_global_runtime_path(self, sim_instance):
        return os.path.join(self.global_runtime, str(sim_instance.uid))

    def sim_global_runtime_stats_path(self, sim_instance):
        return os.path.join(self.sim_global_runtime_path(sim_instance), "runtime.json")

    def sim_global_runtime_config_path(self, sim_instance):
        return os.path.join(self.sim_global_runtime_path(sim_instance), "config.json")

    def sim_global_runtime_workflow_config_path(self, sim_instance):
        return os.path.join(self.sim_global_runtime_path(sim_instance), "workflow_config.json")

    def sim_global_runtime_exec_path(self, sim_instance):
        return os.path.join(self.sim_global_runtime_path(sim_instance), "sim_exec")

    def clean_sim(self, sim_instance):
        sim_instance = self.__retrieve_results_data(sim_instance)
        sim_instance = self.__retrieve_runtime_data(sim_instance)
        sim_instance = self.__retrieve_logs_data(sim_instance)
        return sim_instance

    def clean_sims(self, sim_instances):
        return [self.clean_sim(sim_instance) for sim_instance in sim_instances]

    def __move_folder(self, old, new, description, id):
        if (os.path.isdir(new)):
            self.logger.info("Removing old {} folder for {}".format(description, id))
            shutil.rmtree(new)

        self.logger.info("Moving {} at '{}' to '{}'".format(description, old, new))
        shutil.move(old, new)

    def __remove_folder(self, old, description, id):
        if (os.path.isdir(old)):
            self.logger.info("Removing {} folder for {}".format(description, id))
            shutil.rmtree(old)

    def __move_file(self, old, new, description, id):
        if (os.path.isfile(new)):
            self.logger.info("Removing old {} file for {}".format(description, id))
            os.remove(new)

        self.logger.info("Moving {} at '{}' to '{}'".format(description, old, new))
        shutil.move(old, new)

    def __copy_file(self, old, new, description, id):
        if (os.path.isfile(new)):
            self.logger.info("Removing old {} file for {}".format(description, id))
            os.remove(new)

        self.logger.info("Copying {} at '{}' to '{}'".format(description, old, new))
        shutil.copy2(old, new)

    def __retrieve_results_data(self, sim_instance):
        os.makedirs(self.sim_global_results_path(sim_instance), exist_ok=True)

        sim_local_results_path = sim_instance.local_results_path
        sim_global_results_path = self.sim_global_results_path(sim_instance)

        if self.cnf.tryGet("output_handler", "remove_sca"):
            self.__remove_folder(sim_local_results_path, "results", sim_instance.uid)
        else:
            self.__move_folder(sim_local_results_path, sim_global_results_path, "results", sim_instance.uid)

        return sim_instance


    def __retrieve_logs_data(self, sim_instance):
        os.makedirs(self.sim_global_logs_path(sim_instance), exist_ok=True)

        sim_local_logs_path = sim_instance.local_logs_path
        sim_global_logs_path = self.sim_global_logs_path(sim_instance)

        self.__move_folder(sim_local_logs_path, sim_global_logs_path, "logs", sim_instance.uid)

        return sim_instance


    def __retrieve_runtime_data(self, sim_instance):
        os.makedirs(self.sim_global_runtime_path(sim_instance), exist_ok=True)

        sim_global_runtime_stats_path = self.sim_global_runtime_stats_path(sim_instance)
        sim_instance.set_processed_time()
        self.logger.info("Writing runtime stats data to '{}'".format(sim_global_runtime_stats_path))

        with open(sim_global_runtime_stats_path, "w") as fp:
            json.dump(sim_instance.get_stats_dict(), fp, indent=4)


        sim_local_config_path = sim_instance.local_config_path
        sim_global_config_path = self.sim_global_runtime_config_path(sim_instance)

        sim_global_workflow_config_path = self.sim_global_runtime_workflow_config_path(sim_instance)

        sim_global_exec_path = self.sim_global_runtime_exec_path(sim_instance)

        self.__move_file(sim_local_config_path, sim_global_config_path, "config", sim_instance.uid)
        self.__copy_file(self.config_path, sim_global_workflow_config_path, "workflow config", sim_instance.uid)
        self.__move_file(sim_instance.local_exec_path, sim_global_exec_path, "sim executable", sim_instance.uid)

        return sim_instance

    def output_json(self, file_path, data):
        with open(file_path, "w") as fp:
            json.dump(data, fp, indent=4)

    def output_manager_runtime(self, runtime_stats):
        runtime_path = os.path.join(self.global_runtime, "manager_runtime.json")
        self.output_json(runtime_path, runtime_stats)

    def output_design_point_cache_runtime(self, runtime_stats):
        runtime_path = os.path.join(self.global_runtime, "design_point_cache_runtime.json")
        self.output_json(runtime_path, runtime_stats)

    def output_design_point_queue_runtime(self, id, runtime_stats):
        runtime_path = os.path.join(self.global_runtime, "design_point_queue_{}_runtime.json".format(id))
        self.output_json(runtime_path, runtime_stats)

    def output_resource_controller_runtime(self, runtime_stats):
        runtime_path = os.path.join(self.global_runtime, "resource_controller_runtime.json")
        self.output_json(runtime_path, runtime_stats)

    def output_output_handler_runtime(self, runtime_stats):
        runtime_path = os.path.join(self.global_runtime, "output_handler_runtime.json")
        self.output_json(runtime_path, runtime_stats)

    def get_runtime_stats(self):
        return self.stats.get_stats_dict()

    def shutdown(self):
        # Output your own runtime stats (which are updated based on all actions in shutdown)
        return