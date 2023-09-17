import json
import os
import sys
import time

from multiprocessing import Process
from pathlib import Path, PosixPath
from dask.distributed import LocalCluster, Client, wait
from dask_jobqueue import SLURMCluster

from src.dask_worker import dask_worker
from src.designpointqueue import DesignPointQueue

from src.utils.config_reader import Config
from src.utils.logger import logger
from src.utils.stats import Stats

class ResourceController:
    logger = None
    cnf = None
    logs_path = None
    cluster = None
    client = None
    running_tasks = None
    stats = None

    def __init__(self, config_path, logs_path):
        # TODO:
        # - check if all required config options are available/correct
        self.logs_path = logs_path
        os.makedirs(self.logs_path, exist_ok=True)

        self.stats = Stats()

        self.logger = logger("resource_controller", Path(self.logs_path))

        self.logger.info("Reading in config file: " + str(config_path))
        self.cnf = Config(Path(config_path), Path(self.logs_path), "config_resource_controller")

        num_workers = 1

        if (self.cnf.tryGet("resource_controller")):
            self.logger.info("Resource controller config provided")
            if (self.cnf.tryGet("resource_controller", "cluster")):
                cluster_type = self.cnf.tryGet("resource_controller", "cluster", "interface")
                if (cluster_type == "SLURM"):
                    self.logger.info("Resource controller config specifies supported cluster {}".format(cluster_type))

                    num_jobs = self.cnf.tryGet("resource_controller", "cluster", "jobs")
                    job_cores = self.cnf.tryGet("resource_controller", "cluster", "job_cores")
                    job_processes = self.cnf.tryGet("resource_controller", "cluster", "job_processes")
                    job_memory = self.cnf.tryGet("resource_controller", "cluster", "job_memory")

                    num_workers = num_jobs * job_processes

                    self.cluster = SLURMCluster(job_directives_skip=["--mem"], cores=job_cores, processes=job_processes, memory=job_memory, worker_extra_args=['--resources slots={}'.format(job_cores)], log_directory=self.logs_path)
                    self.cluster.scale(jobs=num_jobs)
                    # TODO: check if number of slots does not exceed num of cores per worker
                    # TODO: wait for all workers to arrive?
                elif (cluster_type == "local"):
                    self.logger.info("Resource controller config specifies supported cluster {}".format(cluster_type))

                    num_workers = self.cnf.tryGet("resource_controller", "cluster", "n_workers")
                    threads_per_worker = self.cnf.tryGet("resource_controller", "cluster", "threads_per_worker")
                    self.cluster = LocalCluster(n_workers=num_workers, threads_per_worker=threads_per_worker, resources={"slots": threads_per_worker})
                else:
                    self.logger.info("Unsupported cluster {} specified in config".format(cluster_type))
                    self.logger.info("Assuming default local dask cluster instead")
                    self.cluster = LocalCluster(n_workers=1, threads_per_worker=6, resources={"slots": 6})
        else:
            self.logger.info("No resource controller config provided")
            self.logger.info("Assuming local dask cluster")
            self.cluster = LocalCluster(n_workers=1, threads_per_worker=6, resources={"slots": 6})

        self.client = Client(self.cluster)

        while ((self.client.status == "running") and (len(self.client.scheduler_info()["workers"]) < num_workers)):
            time.sleep(0.1)

        self.client.forward_logging()
        self.running_tasks = {}

    def __set_sim_instances_time_stat(self, sim_instances, *args):
        sims = []
        for sim_instance in sim_instances:
            sim_instance.record_time_stat(*args)
            sims += [sim_instance]
        return sims

    # Does not wait for sim to complete
    def __submit_task(self, sim_instance):
        self.logger.info("Sending sim instance {} to worker cluster".format(sim_instance.uid))
        sim_instance.record_time_stat("general", "resource_controller_submit")
        future = self.client.submit(dask_worker, sim_instance, resources={"slots": sim_instance.num_slots()}, fifo_timeout="0ms")
        self.running_tasks[sim_instance.uid] = future

    def __get_all_completed(self):
        completed_sim_instances = []
        for i, uid in enumerate(self.running_tasks.keys()):
            # TODO: check for all other statusses
            future = self.running_tasks[uid]
            if (future.status == "finished"):
                self.logger.info("Retrieving results for sim instance: {}".format(uid))
                completed_sim_instances.append(future.result())

        for sim_instance in completed_sim_instances:
            self.running_tasks.pop(sim_instance.uid)

        return completed_sim_instances

    # Does not wait for sims to complete
    def submit_tasks(self, sim_instances):
        self.logger.info("Sending all tasks ({}) in queue to worker cluster".format(len(sim_instances)))
        for sim_instance in sim_instances:
            self.__submit_task(sim_instance)

    # Waits for sims to complete
    def evaluate_tasks(self, sim_instances):
        sim_instances = self.__set_sim_instances_time_stat(sim_instances, "general", "resource_controller_entry")
        self.submit_tasks(sim_instances)
        return self.wait_for_tasks(sim_instances)

    def wait_for_tasks(self, sim_instances):
        # TODO: catch sim instance not in running tasks
        self.logger.info("Retrieving DASK futures for given tasks")
        futures = [self.running_tasks[sim_instance.uid] for sim_instance in sim_instances]
        self.logger.info("Waiting for all tasks to complete")
        wait(futures)
        self.logger.info("All tasks have completed")

        completed_sim_instances = [future.result() for future in futures]
        completed_sim_instances = self.__set_sim_instances_time_stat(completed_sim_instances, "general", "resource_controller_retrieval")
        return completed_sim_instances

    def wait_for_active_tasks(self):
        completed_sim_instances = []
        self.logger.info("Waiting for all active tasks to have completed")

        for uid in self.running_tasks.keys():
            sim_instance = self.running_tasks[uid].result()
            completed_sim_instances.append(sim_instance)
            self.logger.info("Retrieving results for sim instance: {}".format(task["uid"]))

        self.running_tasks = {}

        return completed_sim_instances

    def get_runtime_stats(self):
        return self.stats.get_stats_dict()

    def shutdown(self):
        # TODO:
        # - close dask stuff
        return
