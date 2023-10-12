import os
import json

from src.designpointqueue import DesignPointQueue

def has_dict_keys(dict, keys):
    for key in keys:
        if not key in dict:
            return False
    return True

class WorkflowConfig:
    workflow_config = None

    def __init__(self, sims_path, local_sim_config, local_sim_exec,
                 local_sim_results, local_sim_logs, local_sim_out,
                 global_sim_results, global_sim_logs, global_sim_runtime,
                 design_queues,
                 uid_scheme,
                 cluster_config, remove_sca=False):

        self.workflow_config = {}
        self.workflow_config["sims_path"] = sims_path
        self.workflow_config["local_sim_config"] = local_sim_config
        self.workflow_config["local_sim_exec"] = local_sim_exec
        self.workflow_config["local_sim_results"] = local_sim_results
        self.workflow_config["local_sim_logs"] = local_sim_logs
        self.workflow_config["local_sim_out"] = local_sim_out
        self.workflow_config["global_sim_runtime"] = global_sim_runtime
        self.workflow_config["global_sim_results"] = global_sim_results
        self.workflow_config["global_sim_logs"] = global_sim_logs
        self.workflow_config["uid_scheme"] = uid_scheme
        self.workflow_config["design_point_queues"] = {}
        self.workflow_config["resource_controller"] = {}
        self.workflow_config["output_handler"] = { "remove_sca": remove_sca }

        for design_point_queue_config in design_queues:
            self.add_design_point_queue(design_point_queue_config)

        self.set_cluster_config(cluster_config)

    @staticmethod
    def create_slurm_cluster_config(jobs, job_cores, job_processes, job_memory):
        cluster_config = {}
        cluster_config["interface"] = "SLURM"
        cluster_config["jobs"] = jobs
        cluster_config["job_cores"] = job_cores
        cluster_config["job_processes"] = job_processes
        cluster_config["job_memory"] = job_memory
        return cluster_config

    @staticmethod
    def create_local_cluster_config(n_workers, threads_per_worker):
        cluster_config = {}
        cluster_config["interface"] = "local"
        cluster_config["n_workers"] = n_workers
        cluster_config["threads_per_worker"] = threads_per_worker
        return cluster_config

    @staticmethod
    def __verify_local_cluster_config(cluster_config):
        keys = ["interface", "n_workers", "threads_per_worker"]

        if not has_dict_keys(cluster_config, keys):
            return False

        if cluster_config["interface"] != "local":
            return False

        pos_vals = ["n_workers", "threads_per_worker"]
        if any([cluster_config[key] <= 0 for key in pos_vals]):
            return False

        return True


    @staticmethod
    def __verify_slurm_cluster_config(cluster_config):
        keys = ["interface", "jobs", "job_cores", "job_processes", "job_memory"]

        if not has_dict_keys(cluster_config, keys):
            return False

        if cluster_config["interface"] != "SLURM":
            return False

        pos_vals = ["jobs", "job_cores", "job_processes"]
        if any([cluster_config[key] <= 0 for key in pos_vals]):
            return False

        return True

    @staticmethod
    def __verify_design_point_queue_config(design_point_queue_config):
        keys = ["id", "priority", "sorting_algo"]
        supported_sorting_algos = DesignPointQueue.supported_sorting_algos

        if not has_dict_keys(design_point_queue_config, keys):
            return False

        if not design_point_queue_config["sorting_algo"] in supported_sorting_algos:
            return False

        non_neg_vals = ["priority"]
        if any([design_point_queue_config[key] < 0 for key in non_neg_vals]):
            return False

        return True

    @staticmethod
    def create_design_point_queue_config(id, priority, sorting_algo):
        queue_config = {}
        queue_config["id"] = id
        queue_config["priority"] = priority
        queue_config["sorting_algo"] = sorting_algo

        return queue_config

    def add_design_point_queue(self, design_point_queue_config):
            if not self.__verify_design_point_queue_config(design_point_queue_config):
                raise Exception("Invalid design point queue config")

            id = design_point_queue_config["id"]
            priority = design_point_queue_config["priority"]
            sorting_algo = design_point_queue_config["sorting_algo"]

            queue_config = {}
            queue_config["priority"] = priority
            queue_config["sorting_algo"] = sorting_algo
            self.workflow_config["design_point_queues"][id] = queue_config

    def set_cluster_config(self, cluster_config):
        if "interface" in cluster_config:
            interface = cluster_config["interface"]
        else:
            Exception("Invalid cluster configuration")

        if interface == "local":
            if not self.__verify_local_cluster_config(cluster_config):
                raise Exception("Invalid local cluster configuration")
        elif interface == "SLURM":
            if not self.__verify_slurm_cluster_config(cluster_config):
                raise Exception("Invalid SLURM cluster configuration")
        else:
            raise Exception("Unsupported interface for config")

        self.workflow_config["resource_controller"]["cluster"] = cluster_config

    def conf(self):
        return self.workflow_config

    def write_conf(self, file_path):
        config = self.conf()

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as fp:
            json.dump(config, fp, indent=4)




class SiminstanceConfig:
    sim_instance_config = None
    supported_simulators = ["omnet"]

    def __init__(self, simulator, simulator_config):
        self.sim_instance_config = {}

        if simulator in SiminstanceConfig.supported_simulators:
            self.sim_instance_config["simulator"] = simulator
            self.sim_instance_config[simulator] = simulator_config.conf()
        else:
            raise Exception("Unsupported simulator")

    def conf(self):
        return self.sim_instance_config

    def write_conf(self, file_path):
        config = self.conf()

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as fp:
            json.dump(config, fp, indent=4)


class OmnetSimConfig:
    omnet_config = None

    def __init__(self, opp_makemake_force, make_verbose,
                 ini_file, model_config, environment,
                 time_limit=None, pdes=None, record_eventlog=False,
                 libraries=None, ned_paths=None,
                 metadata=None):

        self.omnet_config = {}
        # self.omnet_config["executable_filename"] = workflow_config["local_sim_exec"]
        self.omnet_config["make"] = {}
        self.omnet_config["simulation"] = {}

        makemake_config = {}
        makemake_config["force"] = opp_makemake_force
        # makemake_config["out_path"] = workflow_config["local_sim_out"]

        self.omnet_config["make"]["opp_makemake"] = makemake_config
        self.omnet_config["make"]["verbose"] = make_verbose

        self.omnet_config["simulation"]["ini"] = ini_file
        self.omnet_config["simulation"]["config"] = model_config
        self.omnet_config["simulation"]["environment"] = environment
        # self.omnet_config["simulation"]["results_path"] = workflow_config["local_sim_results"]

        if libraries:
            self.omnet_config["simulation"]["libraries"] = libraries

        if ned_paths:
            self.omnet_config["simulation"]["ned_paths"] = ned_paths

        if time_limit:
            self.set_time_limit(time_limit["type"], time_limit["duration"])

        if pdes:
            num_lps = pdes["num_lps"]
            self.omnet_config["simulation"]["pdes"] = {"num_lps": num_lps}

        self.omnet_config["simulation"]["record_eventlog"] = record_eventlog

        self.omnet_config["metadata"] = metadata

    def set_time_limit(self, type, duration):
        time_limit_config = {"limit_type": type, "duration": duration}
        self.omnet_config["simulation"]["time_limit"] = time_limit_config

    def set_pdes(self, num_lps):
        self.omnet_config["simulation"]["pdes"] = {"num_lps": num_lps}

    def conf(self):
        return self.omnet_config

    def write_conf(self, file_path):
        config = self.conf()

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as fp:
            json.dump(config, fp, indent=4)