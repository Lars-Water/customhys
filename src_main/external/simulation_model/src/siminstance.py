import subprocess
import json
import os
import time

from pathlib import Path, PosixPath

from src.utils.config_reader import Config
from src.utils.logger import logger
from src.utils.hash import md5_dir
from src.utils.stats import Stats

import uuid


class Siminstance:
    logger = None
    cnf = None
    uid = None
    path = None
    local_logs_path = None
    local_config_path = None
    workflow_config = None
    hash = None

    stats = None

    def __init__(self, sim_path, workflow_config):
        self.stats = Stats()
        self.stats.record_time_stat("general", "creation_time")

        self.path = sim_path
        self.workflow_config = workflow_config

        self.uid = self.calc_uid(self.path, self.workflow_config["uid_scheme"])
        self.stats.record_stat(self.uid, "general", "UID")

        self.local_logs = self.workflow_config["local_sim_logs"]
        self.local_logs_path = os.path.join(self.path, self.local_logs)

        os.makedirs(self.local_logs_path, exist_ok=True)

        self.logger = logger("siminstance_" + str(self.uid), Path(self.local_logs_path))
        self.logger.info("Path to simulation instance source: " + str(self.path))
        self.logger.info("UID of sim instance: {}".format(self.uid))
        self.logger.info("Local logs path: {}".format(self.local_logs))

        self.local_results = self.workflow_config["local_sim_results"]
        self.local_results_path = os.path.join(self.path, self.local_results)
        self.logger.info("Local results path: {}".format(self.local_results_path))

        self.local_exec = self.workflow_config["local_sim_exec"]
        self.local_exec_path = os.path.join(self.path, self.local_exec)
        self.logger.info("Local executable name: {}".format(self.local_exec))

        self.local_config = self.workflow_config["local_sim_config"]
        self.local_config_path = os.path.join(self.path, self.local_config + ".json")
        self.logger.info("Local config path: {}".format(self.local_config_path))

        self.local_out = self.workflow_config["local_sim_out"]
        self.local_out_path = os.path.join(self.path, self.local_out)
        self.logger.info("Local out path: {}".format(self.local_out_path))

        self.logger.info("Reading in config file: " + str(self.local_config_path))
        self.cnf = Config(Path(self.local_config_path), Path(self.local_logs_path), "config_siminstance_" + str(self.uid))

        # TODO: Change this to a configuration option.
        # NOTE: Code added by Lars. This is a temporary solution to the problem of the output being in a non-standard format.
        self.transform_scalar_output = True

    @staticmethod
    def calc_uid(sim_path, uid_scheme):
        if uid_scheme == "md5-files":
            return md5_dir(sim_path)
        if uid_scheme == "uuid":
            return str(uuid.uuid4())
        else:
            raise Exception("Received unknown uid scheme")

    def __create_omnet_make_make_command(self):
        make_make_command = ["opp_makemake"]

        if (self.local_exec):
            self.logger.info("opp_makemake will use {} as executable filename".format(self.local_exec))
            make_make_command += ["-o", self.local_exec]

        if (self.cnf.tryGet("omnet", "make")):
            self.logger.info("Omnet config has specification for make")

            if (self.cnf.tryGet("omnet", "make", "opp_makemake")):
                self.logger.info("Omnet config has specification for opp_makemake")

                if (self.cnf.tryGet("omnet", "make", "opp_makemake", "force")):
                    self.logger.info("opp_makemake will force overwrite of makefile")
                    make_make_command += ["-f"]

                # TODO: make sure path exists
                if (self.local_out_path):
                    self.logger.info("opp_makemake will output executable folder {}".format(self.local_out_path))
                    make_make_command += ["-O", self.local_out_path]
        else:
            self.logger.warn("Omnet config has no specification for opp_makemake")
            self.logger.info("Defaulting to: 'opp_makemake -f'")
            make_make_command += ["-f"]

        self.logger.debug("MakeMake command:\n"+ " ".join(make_make_command))
        return make_make_command

    def __create_omnet_make_command(self):
        make_command = ["make"]

        if (self.cnf.tryGet("omnet", "make")):
            self.logger.info("Omnet config has specification for make")
            if (self.cnf.tryGet("omnet", "make", "verbose")):
                self.logger.info("Make output will be verbose")
                make_command += ["V=1"]
            if (self.cnf.tryGet("omnet", "make", "ignoreWarnings")):
                self.logger.info("Adding following ignore warnings to make command: " + ", ".join(self.cnf.tryGet("omnet", "make", "ignoreWarnings")))
                for warning in self.cnf.tryGet("omnet", "make", "ignoreWarnings"):
                    make_command += [warning]
        else:
            self.logger.warn("Omnet config has no specification for make")
            self.logger.info("Defaulting to: 'make'")

        self.logger.info("Make command:\n"+ " ".join(make_command))
        return make_command


    def __create_omnet_sim_command(self):
        simulation_command = []

        if (self.cnf.tryGet("omnet", "simulation", "param_sweep", "active")):
            self.logger.info("Omnet execution will make use of param_sweep")
            # TODO: integrate usage of param_sweep functionality
        else:
            self.logger.info("Omnet execution will not make use of param_sweep (i.e., it will be sequential)")

        if (self.local_exec):
            simulation_command += ["./" + self.local_exec]
            self.logger.info("Omnet executable is: {}".format(self.local_exec))
        else:
            self.logger.warn("Omnet executable not defined")
            raise Exception("No executable was defined in the configuration")

        if (self.cnf.tryGet("omnet", "simulation", "environment")):
            simulation_command += ["-u", self.cnf.tryGet("omnet", "simulation", "environment")]
            self.logger.info("Omnet environment is: {}".format(self.cnf.tryGet("omnet", "simulation", "environment")))
        else:
            self.logger.warn("Omnet environment not defined")
            raise Exception("No omnet environment was defined in the configuration")

        if (self.local_results_path):
            simulation_command += ["--result-dir", self.local_results_path]
            self.logger.info("Omnet result output path is: {}".format(self.local_results_path))
        else:
            self.logger.warn("Omnet result output path not defined")
            raise Exception("No omnet result output path was defined in the configuration")

        if (self.cnf.tryGet("omnet", "simulation", "time_limit")):
            time_limit_type = self.cnf.tryGet("omnet", "simulation", "time_limit")["limit_type"]
            time_limit_duration = self.cnf.tryGet("omnet", "simulation","time_limit")["duration"]

            if (time_limit_type == "real-time"):
                self.logger.info("Omnet will run with time limit type {} for duration of {}".format(time_limit_type, time_limit_duration))
                simulation_command += ["--real-time-limit", time_limit_duration]
            else:
                # TODO: add support for other time limits
                pass

        if (self.cnf.tryGet("omnet", "simulation", "ned_paths")):
            ned_paths = self.cnf.tryGet("omnet", "simulation", "ned_paths")
            ned_paths += [self.path]
            simulation_command += ["-n", "; ".join(ned_paths)]

        if (self.cnf.tryGet("omnet", "simulation", "libraries")):
            lib_paths = self.cnf.tryGet("omnet", "simulation", "libraries")
            for lib_path in lib_paths:
                simulation_command += ["-l", lib_path]

        if (self.cnf.tryGet("omnet", "simulation", "record_eventlog")):
            self.logger.info("Omnet will record eventlog")
            simulation_command += ["--record-eventlog"]

        # # Todo fix True for other experiments
        # if (self.cnf.tryGet("omnet", "simulation", "oversubscribe") or True):
        #     self.logger.info("Omnet will oversubscribe")
        #     simulation_command += ["--oversubscribe"]

        if (self.cnf.tryGet("omnet", "simulation", "pdes")):
            num_lps = self.cnf.tryGet("omnet", "simulation", "pdes", "num_lps")
            self.logger.info("Omnet execution will perform pdes with {} lps".format(num_lps))
            simulation_command = ["mpiexec", "--oversubscribe", "-n",  str(num_lps)] + simulation_command + ["--parallel-simulation", "true", "--parsim-communications-class", "cMPICommunications"]
        else:
            self.logger.info("Omnet execution will not make use of pdes (i.e., it will be sequential)")



        if (self.cnf.tryGet("omnet", "simulation", "config")):
            simulation_command += ["-c", self.cnf.tryGet("omnet", "simulation", "config")]
            self.logger.info("Omnet config is: {}".format(self.cnf.tryGet("omnet", "simulation", "config")))
        else:
            self.logger.warn("Omnet config not defined")
            raise Exception("No omnet config was defined in the configuration")

        if self.cnf.tryGet("omnet", "simulation", "ini"):
            ini = self.cnf.tryGet("omnet", "simulation", "ini")
            simulation_command += [os.path.join(self.path, ini)]

        self.logger.debug("Simulation command:\n"+ " ".join(simulation_command))
        return simulation_command


    def record_total_time_stat(self, args_total, args_start, args_end):
        self.stats.record_time_stat(*args_end)
        total_time = self.stats.get_stat(*args_end) - self.stats.get_stat(*args_start)
        self.stats.record_stat(total_time, *args_total)

    def record_time_stat(self, *args):
        self.stats.record_time_stat(*args)

    def record_stat(self, val, *args):
        self.stats.record_stat(val, *args)

    def get_stats_dict(self):
        return self.stats.get_stats_dict()

    def num_slots(self):
        if self.cnf.tryGet("simulator") == "omnet":
            if self.cnf.tryGet("omnet", "simulation", "pdes"):
                return self.cnf.tryGet("omnet", "simulation", "pdes", "num_lps")
            else:
                return 1
        else:
            return 1

    def set_processed_time(self):
        self.stats.record_time_stat("general", "finish_time")
        total_time = self.stats.get_stat("general", "finish_time") - self.stats.get_stat("general", "creation_time")
        self.stats.record_stat(total_time, "general", "processed_time")

    def compile(self):
        # TODO:
        # - check if commands exist
        # - handle exceptions (read output etc.)
        self.stats.record_time_stat("compilation", "compile_start_time")

        if (self.cnf.tryGet("omnet")):
            self.logger.info("Omnet config data present in config file")

            self.stats.record_time_stat("compilation", "make_make_build_start")
            make_make_command = self.__create_omnet_make_make_command()
            self.record_total_time_stat(("compilation", "make_make_build_time"), ("compilation", "make_make_build_start"), ("compilation", "make_make_build_end"))

            self.logger.info("Running opp_makemake command")

            self.stats.record_time_stat("compilation", "make_make_exec_start")
            make_make_output = subprocess.run(make_make_command, cwd=self.path, capture_output=True)
            self.record_total_time_stat(("compilation", "make_make_exec_time"), ("compilation", "make_make_exec_start"), ("compilation", "make_make_exec_end"))

            self.stats.record_time_stat("compilation", "make_make_store_start")
            self.string_to_file(make_make_output.stdout.decode("utf-8"), self.local_logs_path, "siminstance_{}_opp_makemake_stdout".format(self.uid))
            self.string_to_file(make_make_output.stderr.decode("utf-8"), self.local_logs_path, "siminstance_{}_opp_makemake_stderr".format(self.uid))
            self.record_total_time_stat(("compilation", "make_make_store_time"), ("compilation", "make_make_store_start"), ("compilation", "make_make_store_end"))

            self.logger.info("Opp_makemake execution return code: {}".format(make_make_output.returncode))

            if (make_make_output.returncode == 0):
                self.logger.info("Opp_makemake execution was successfull")
            else:
                self.logger.warn("Opp_makemake execution was not successfull")
                raise Exception("Opp_makemake execution was not successfull")

            self.stats.record_time_stat("compilation", "make_build_start")
            make_command = self.__create_omnet_make_command()
            self.record_total_time_stat(("compilation", "make_build_time"), ("compilation", "make_build_start"), ("compilation", "make_build_end"))


            self.logger.info("Running make command: {}".format(make_command))

            self.stats.record_time_stat("compilation", "make_exec_start")
            make_output = subprocess.run(make_command, cwd=self.path, capture_output=True)
            self.record_total_time_stat(("compilation", "make_exec_time"), ("compilation", "make_exec_start"), ("compilation", "make_exec_end"))

            self.stats.record_time_stat("compilation", "make_store_start")
            self.string_to_file(make_output.stdout.decode("utf-8"), self.local_logs_path, "siminstance_{}_make_stdout".format(self.uid))
            self.string_to_file(make_output.stderr.decode("utf-8"), self.local_logs_path, "siminstance_{}_make_stderr".format(self.uid))
            self.record_total_time_stat(("compilation", "make_store_time"), ("compilation", "make_store_start"), ("compilation", "make_store_end"))

            self.logger.info("Make execution return code: {}".format(make_output.returncode))

            if (make_output.returncode == 0):
                self.logger.info("Make execution was successfull")
            else:
                self.logger.error("Make execution was not successfull")
                self.logger.error(make_output)
                raise Exception("Make execution was not successfull", make_output)


        else:
            self.logger.warn("No supported simulator configuration data was found.")
            raise Exception("No supported simulator configuration data was found.")

        self.record_total_time_stat(("compilation", "compile_time"), ("compilation", "compile_start_time"), ("compilation", "compile_end_time"))

        self.logger.info("Compilation took {:.4} seconds".format(self.stats.get_stat("compilation", "compile_time")))

    def string_to_file(self, output, path, filename):
        with open(os.path.join(path, filename), 'w') as f:
            f.write(output)

    def setCacheHash(self, hash):
        self.hash = hash

    def getCacheHash(self):
        return self.hash


    def run(self):
        # TODO:
        # - Make sure compilation has been performed
        # - Check if executable exists
        # - handle exceptions (read output etc.)
        self.stats.record_time_stat("simulation", "simulation_start")

        if (self.cnf.tryGet("omnet")):
            self.logger.info("Omnet config data present in config file")

            simulation_command = []

            if (self.cnf.tryGet("omnet", "simulation")):
                self.logger.info("Omnet config has specification for simulation execution")

                self.stats.record_time_stat("simulation", "sim_build_start")
                simulation_command = self.__create_omnet_sim_command()
                self.record_total_time_stat(("simulation", "sim_build_time"), ("simulation", "sim_build_start"), ("simulation", "sim_build_end"))

            else:
                self.logger.warn("No supported simulation execution configuration data was found.")
                raise Exception("No supported simulation execution configuration data was found.")

            self.logger.info("Running simulation command: {}".format(simulation_command))

            self.stats.record_time_stat("simulation", "sim_exec_start")
            simulation_output = subprocess.run(simulation_command, cwd=self.path, capture_output=True)
            self.record_total_time_stat(("simulation", "sim_exec_time"), ("simulation", "sim_exec_start"), ("simulation", "sim_exec_end"))

            self.stats.record_time_stat("simulation", "sim_store_start")
            self.string_to_file(simulation_output.stdout.decode("utf-8"), self.local_logs_path, "siminstance_{}_sim_stdout".format(self.uid))
            self.string_to_file(simulation_output.stderr.decode("utf-8"), self.local_logs_path, "siminstance_{}_sim_stderr".format(self.uid))
            self.record_total_time_stat(("simulation", "sim_store_time"), ("simulation", "sim_store_start"), ("simulation", "sim_store_end"))

            self.logger.info("Simulation execution return code: {}".format(simulation_output.returncode))

            if (simulation_output.returncode == 0):
                self.logger.info("Simulation execution was successfull")
            else:
                self.logger.error("Simulation execution was not successfull")
                self.logger.error(simulation_output)
                raise Exception("Simulation execution was not successfull",simulation_output)

        else:
            self.logger.warn("No supported simulator configuration data was found.")
            raise Exception("No supported simulator configuration data was found.")

        self.record_total_time_stat(("simulation", "simulation_time"), ("simulation", "simulation_start"), ("simulation", "simulation_end"))
        self.logger.info("Simulation took {:.4} seconds".format(self.stats.get_stat("simulation", "simulation_time")))

        # NOTE: Custom code by Lars. This is a temporary solution to the problem of the output being in a non-standard format.
        if self.transform_scalar_output:
            self.logger.info("Transforming scalar output to CSV-R format")

            scavetool_command = ["opp_scavetool", "export", "-F", "CSV-R", "-o", "x.csv", "*.sca"]
            conversion_output = subprocess.run(scavetool_command, cwd=self.local_results_path, capture_output=True)
            self.string_to_file(conversion_output.stdout.decode("utf-8"), self.local_logs_path, "siminstance_{}_output_converion_stdout".format(self.uid))
            self.string_to_file(conversion_output.stderr.decode("utf-8"), self.local_logs_path, "siminstance_{}_output_converion_stderr".format(self.uid))

            # Remove all .sca files in the local results path.
            for file in os.listdir(self.local_results_path):
                if file.endswith(".sca"):
                    os.remove(os.path.join(self.local_results_path, file))
