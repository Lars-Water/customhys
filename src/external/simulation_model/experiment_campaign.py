import argparse
import shutil
import os
import json
import random
import time

from experiments import create_sim, create_sim_inet_lans, create_sim_pdes_comm, create_sim_pdes_comp, create_sim_pdes_sub
from experiments import create_sim_seq_comm, create_sim_seq_comp, create_sim_seq_sub
from experiments import create_sim_custom_dummy, create_tictoc


from src.manager import Manager
from src.siminstance import Siminstance
from src.outputhandler import OutputHandler
from src.utils.config_creator import OmnetSimConfig, SiminstanceConfig, WorkflowConfig

def campaign(args):
    dummy_sim_path = args.dummy_path
    inet_base = args.inet_path
    design_queues = [WorkflowConfig.create_design_point_queue_config("base", 0, "FIFO")]
    sims_path = args.sims_path


    if args.platform == "DAS":
        num_nodes = args.num_nodes
        num_workers = args.num_workers
        if args.hyper:
            cluster_config = WorkflowConfig.create_slurm_cluster_config(num_nodes, 64, num_workers, "64GB")
        else:
            cluster_config = WorkflowConfig.create_slurm_cluster_config(num_nodes, 32, num_workers, "64GB")
    elif args.platform == "local":
        num_nodes = 1
        num_workers = args.num_workers
        num_threads_per_worker = int(6 / num_workers)
        cluster_config = WorkflowConfig.create_local_cluster_config(num_workers, num_threads_per_worker)

    workflow_config_file = os.path.join(args.data_path, "config.json")
    workflow_results_folder = os.path.join(args.data_path, "results")
    workflow_logs_folder = os.path.join(args.data_path, "logs")
    workflow_runtime_folder = os.path.join(args.data_path, "runtime")


    workflow_config = WorkflowConfig(sims_path, "config", "run_sim", "results", "logs", "out",
                            workflow_results_folder, workflow_logs_folder, workflow_runtime_folder, design_queues, "md5-files", cluster_config)
    config = workflow_config.conf()
    workflow_config.write_conf(workflow_config_file)

    if args.model == "inet_lans":
        num_sims = args.num_sims
        sim_instances = [create_sim_inet_lans(config, dummy_sim_path, id, inet_base) for id in range(num_sims)]
    elif args.model == "pdes_comm":
        num_sims = args.num_sims
        sim_instances = [create_sim_pdes_comm(config, dummy_sim_path, id, inet_base) for id in range(num_sims)]
    elif args.model == "pdes_comp":
        num_sims = args.num_sims
        sim_instances = [create_sim_pdes_comp(config, dummy_sim_path, id, inet_base) for id in range(num_sims)]
    elif args.model == "pdes_sub":
        num_sims = args.num_sims
        sim_instances = [create_sim_pdes_sub(config, dummy_sim_path, id, inet_base) for id in range(num_sims)]
    elif args.model == "seq_comm":
        num_sims = args.num_sims
        sim_instances = [create_sim_seq_comm(config, dummy_sim_path, id, inet_base) for id in range(num_sims)]
    elif args.model == "seq_comp":
        num_sims = args.num_sims
        sim_instances = [create_sim_seq_comp(config, dummy_sim_path, id, inet_base) for id in range(num_sims)]
    elif args.model == "seq_sub":
        num_sims = args.num_sims
        sim_instances = [create_sim_seq_sub(config, dummy_sim_path, id, inet_base) for id in range(num_sims)]
# -------------------------------------------------------------------------------------------
# NOTE: The following functions are added by me (Lars) and does not exist in the original code.
# -------------------------------------------------------------------------------------------
    elif args.model == "tictoc":
        num_sims = args.num_sims
        sim_instances = [create_tictoc(config, dummy_sim_path, id, inet_base) for id in range(num_sims)]
    elif args.model == "custom":
        num_sims = args.num_sims
        sim_instances = [create_sim_custom_dummy(config, dummy_sim_path, id, inet_base) for id in range(num_sims)]
# -------------------------------------------------------------------------------------------
    else:
        sim_instances = []

    m = Manager(workflow_config_file, workflow_logs_folder)

    m.enqueue_tasks(sim_instances)
    evaluated_sim_instances = m.evaluate_all()
    m.shutdown()

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_path", default="project/", help="specifies the base path of the project")
    arg_parser.add_argument("--inet_path", default="inet4.5/", help="specifies the base path of inet")
    arg_parser.add_argument("--sims_path", default="sims/", help="specifies the path where the sims will be stored")
    arg_parser.add_argument("--dummy_path", default="sims/", help="specifies the path to the dummy sims")
    arg_parser.add_argument("--platform", default="DAS", help="specifies the platform where it will be running")
    arg_parser.add_argument("--model", default="inet_lans", help="specifies the model for which a sequential sim will be performed")
    arg_parser.add_argument("--num_nodes", default=1, help="number of nodes to use in cluster", type=int)
    arg_parser.add_argument("--num_workers", default=1, help="number of workers per node to use in cluster", type=int)
    arg_parser.add_argument("--num_sims", default=6, help="number of sims to use in cluster", type=int)
    arg_parser.add_argument("--hyper", default=False, help="whether hyperthreading", action='store_true')

    args = arg_parser.parse_args()

    campaign(args)



if __name__ == '__main__':
    main()