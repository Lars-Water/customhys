import json
import os
import sys
import time
import threading

from multiprocessing import Process
from pathlib import Path, PosixPath
from dask.distributed import LocalCluster, Client, get_client, wait
from dask_jobqueue import SLURMCluster

from src.dask_worker import dask_worker
from src.designpointqueue import DesignPointQueue

from src.utils.config_reader import Config
from src_main.tools.logger import logger
from src.utils.stats import Stats

from src_main.tools.local_parallelization.rpc import requires_main_process, callable_from_main, managed_by_main_process


@managed_by_main_process
class ResourceController:
    logger = None
    cnf = None
    logs_path = None
    cluster = None
    client = None
    running_tasks = None
    stats = None
    _client_lock = None
    scheduler_address = None
    main_pid = None

    def __init__(self, config_path, logs_path):
        # TODO:
        # - check if all required config options are available/correct
        self.logs_path = logs_path
        os.makedirs(self.logs_path, exist_ok=True)
        self._client_lock = threading.Lock()
        self.main_pid = os.getpid()

        self.stats = Stats()

        self.logger = logger("resource_controller", Path(self.logs_path), disabled=False)

        self.logger.info("Reading in config file: " + str(config_path))
        self.logger.setLevel("DEBUG")
        self.cnf = Config(Path(config_path), Path(self.logs_path), "config_resource_controller")

        self.num_workers = 1

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
                    walltime =  self.cnf.tryGet("resource_controller", "cluster", "walltime")

                    self.num_workers = num_jobs * job_processes

                    self.cluster = SLURMCluster(job_directives_skip=["--mem"], cores=job_cores, processes=job_processes, walltime=walltime, death_timeout=5*60*60, memory=job_memory, worker_extra_args=['--resources slots={}'.format(job_cores)], log_directory=self.logs_path)
                    self.cluster.scale(jobs=num_jobs)
                    
                elif (cluster_type == "local"):
                    self.logger.info("Resource controller config specifies supported cluster {}".format(cluster_type))

                    self.num_workers = self.cnf.tryGet("resource_controller", "cluster", "n_workers")
                    threads_per_worker = self.cnf.tryGet("resource_controller", "cluster", "threads_per_worker")
                    self.cluster = LocalCluster(n_workers=self.num_workers, threads_per_worker=threads_per_worker, resources={"slots": threads_per_worker})
                else:
                    self.logger.info("Unsupported cluster {} specified in config".format(cluster_type))
                    self.logger.info("Assuming default local dask cluster instead")
                    self.cluster = LocalCluster(n_workers=1, threads_per_worker=6, resources={"slots": 6})
        else:
            self.logger.info("No resource controller config provided")
            self.logger.info("Assuming local dask cluster")
            self.cluster = LocalCluster(n_workers=1, threads_per_worker=6, resources={"slots": 6})

        self.scheduler_address = self.cluster.scheduler_address
        self.logger.info(f"Dask scheduler running at: {self.scheduler_address}")

        self.running_tasks = {}
        self.running_tasks_slow = {}

    def get_client(self):
        with self._client_lock:
            if self.client is None:
                self.logger.info(f"Creating Dask client for process {os.getpid()} connecting to {self.scheduler_address}")
                self.client = Client(self.scheduler_address, timeout=60, set_as_default=True)
                self.logger.info(f"Waiting for workers...")
                self.client.wait_for_workers(self.num_workers)
                self.logger.info(f"Client connected: {self.client}")
                self.logger.info(f"Scheduler info: {self.client.scheduler_info()}")
                
                # try:
                #     self.client.forward_logging()
                # except Exception as e:
                #     self.logger.error(f"Error while client.forward_logging(): {e}. It is disabled.")

        return self.client

    def get_running_tasks(self, uids):
        running_tasks = {}
        for uid in uids:
            if uid in self.running_tasks:
                running_tasks[uid] = self.running_tasks[uid]
            elif uid in self.running_tasks_slow:
                running_tasks[uid] = self.running_tasks_slow[uid]
        return running_tasks

    def __set_sim_instances_time_stat(self, sim_instances, *args):
        sims = []
        for sim_instance in sim_instances:
            sim_instance.record_time_stat(*args)
            sims += [sim_instance]
        return sims

    # Does not wait for sim to complete
    def __submit_task(self, sim_instance):
        self.logger.debug("Sending sim instance {} to worker cluster".format(sim_instance.uid))
        sim_instance.record_time_stat("general", "resource_controller_submit")
        client = self.get_client()
        with client.as_current():
            self.logger.info(f"Client.current(): {Client.current()}")
            self.logger.info(f"Client.current().scheduler_info(): {Client.current().scheduler_info()}")
            future = Client.current().submit(dask_worker, sim_instance, resources={"slots": sim_instance.num_slots()}, fifo_timeout="50ms")
        self.logger.info(f"future: {future}")
        self.running_tasks[sim_instance.uid] = future

    def __get_all_completed(self):
        completed_sim_instances = []
        for i, uid in enumerate(self.running_tasks.keys()):
            # TODO: check for all other statusses
            future = self.running_tasks[uid]
            if (future.status == "finished"):
                self.logger.debug("Retrieving results for sim instance: {}".format(uid))
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

    def get_futures_from_sim_instances(self, sim_instances):
        return [self.running_tasks[sim_instance.uid] for sim_instance in sim_instances]

    def wait_for_tasks(self, sim_instances):
        # TODO: catch sim instance not in running tasks
        self.get_client()  # Ensure client is initialized
        futures = self.get_futures_from_sim_instances(sim_instances)
        wait(futures)

        completed_sim_instances = [future.result() for future in futures]
        completed_sim_instances = self.__set_sim_instances_time_stat(completed_sim_instances, "general", "resource_controller_retrieval")
        return completed_sim_instances

    def wait_for_active_tasks(self):
        completed_sim_instances = []
        self.get_client() # Ensure client is initialized
        self.logger.debug("Waiting for all active tasks to have completed")

        for uid in list(self.running_tasks.keys()):
            sim_instance = self.running_tasks[uid].result()
            completed_sim_instances.append(sim_instance)
            self.logger.debug("Retrieving results for sim instance: {}".format(uid))

        self.running_tasks = {}

        return completed_sim_instances

    def get_runtime_stats(self):
        return self.stats.get_stats_dict()

    def shutdown(self):
        self.logger.info("Shutting down ResourceController.")
        if self.client:
            self.client.close()
        if self.cluster:
            # Only the main process should close the cluster
            if os.getpid() == self.main_pid:
                self.cluster.close()
        self.logger.info("ResourceController shutdown complete.")
        return
