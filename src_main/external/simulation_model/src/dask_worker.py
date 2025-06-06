from time import sleep
import logging
import time
import sys
import os
from src_main.tools.logger import setLevelLogger
from dask.distributed import get_worker
import subprocess


def dask_worker(sim_instance):
    sim_instance.record_time_stat("general", "dask_worker_start")
    sim_instance.logger.setLevel(logging.DEBUG)
    worker_id = get_worker().id
    sim_instance.record_stat(worker_id, "general", "dask_worker")
    sim_instance.compile()
    sim_instance.run()
    sim_instance.record_total_time_stat(("general", "dask_worker_time"), ("general", "dask_worker_start"), ("general", "dask_worker_end"))

    # sim_instance.transform_scalar_output()

    return sim_instance
