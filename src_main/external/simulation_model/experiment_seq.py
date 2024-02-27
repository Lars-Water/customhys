import argparse
import shutil
import os
import json
import random
import time

from experiments import create_sim, create_sim_inet_lans, create_sim_pdes_comm, create_sim_pdes_comp, create_sim_pdes_sub
from experiments import create_sim_seq_comm, create_sim_seq_comp, create_sim_seq_sub

from src.siminstance import Siminstance
from src.outputhandler import OutputHandler
from src.utils.config_creator import OmnetSimConfig, SiminstanceConfig, WorkflowConfig

def sequential_baseline(args, id=0):
    dummy_sim_path = args.dummy_path
    inet_base = args.inet_path
    sims_path = args.sims_path

    if args.multi:
        id = random.randint(0, 100000)
        workflow_config_file = os.path.join(args.data_path, str(id), "config.json")
        workflow_results_folder = os.path.join(args.data_path, str(id), "results")
        workflow_logs_folder = os.path.join(args.data_path, str(id), "logs")
        workflow_runtime_folder = os.path.join(args.data_path, str(id), "runtime")
    else:
        workflow_config_file = os.path.join(args.data_path, "config.json")
        workflow_results_folder = os.path.join(args.data_path, "results")
        workflow_logs_folder = os.path.join(args.data_path, "logs")
        workflow_runtime_folder = os.path.join(args.data_path, "runtime")

    if args.platform == "DAS":
        cluster_config = WorkflowConfig.create_slurm_cluster_config(1, 32, 1, "64GB")
    elif args.platform == "local":
        cluster_config = WorkflowConfig.create_local_cluster_config(1, 6)

    design_queues = [WorkflowConfig.create_design_point_queue_config("base", 0, "FIFO")]

    workflow_config = WorkflowConfig(sims_path, "config", "run_sim", "results", "logs", "out",
                            workflow_results_folder, workflow_logs_folder, workflow_runtime_folder, design_queues, "md5-files", cluster_config)
    config = workflow_config.conf()
    workflow_config.write_conf(workflow_config_file)

    oh = OutputHandler(workflow_config_file, workflow_logs_folder)

    if args.model == "inet_lans":
        sim_instance = create_sim_inet_lans(config, dummy_sim_path, id, inet_base)
    elif args.model == "pdes_comm":
        sim_instance = create_sim_pdes_comm(config, dummy_sim_path, id, inet_base)
    elif args.model == "pdes_comp":
        sim_instance = create_sim_pdes_comp(config, dummy_sim_path, id, inet_base)
    elif args.model == "pdes_sub":
        sim_instance = create_sim_pdes_sub(config, dummy_sim_path, id, inet_base)
    elif args.model == "seq_comm":
        sim_instance = create_sim_seq_comm(config, dummy_sim_path, id, inet_base)
    elif args.model == "seq_comp":
        sim_instance = create_sim_seq_comp(config, dummy_sim_path, id, inet_base)
    elif args.model == "seq_sub":
        sim_instance = create_sim_seq_sub(config, dummy_sim_path, id, inet_base)
    else:
        sim_instance = create_sim_inet_lans(config, dummy_sim_path, id, inet_base)

    sim_instance.compile()
    sim_instance.run()

    oh.clean_sim(sim_instance)

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_path", default="experiments/", help="specifies the base path of the project")
    arg_parser.add_argument("--inet_path", default="inet4.5/", help="specifies the base path of inet")
    arg_parser.add_argument("--sims_path", default="sims/", help="specifies the path where the sims will be stored")
    arg_parser.add_argument("--dummy_path", default="sims/", help="specifies the path to the dummy sims")
    arg_parser.add_argument("--platform", default="DAS", help="specifies the platform where it will be running")
    arg_parser.add_argument("--model", default="inet", help="specifies the model for which a sequential sim will be performed")
    arg_parser.add_argument("--multi", default=False, help="whether multi", action='store_true')

    args = arg_parser.parse_args()

    sequential_baseline(args)



if __name__ == '__main__':
    main()