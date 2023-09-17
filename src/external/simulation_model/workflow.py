import argparse
import shutil
import os
import json

from src.manager import Manager
from src.resourcecontroller import ResourceController
from src.outputhandler import OutputHandler
from src.siminstance import Siminstance
from src.utils.config_creator import OmnetSimConfig, SiminstanceConfig, WorkflowConfig

from create_sims import create_sim_seq_comm, create_sim_pdes_comm, create_sim_custom_dummy

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--workflowConfigFile", default="workflow.json", help="specifies the path to the .json file that contains the general configuration of the workflow")
    arg_parser.add_argument("--workflowLogsFolder", default="logs/", help="specifies the path to the folder that will contain the logs of the (components in the) workflow")
    arg_parser.add_argument("--workflowRuntimeFolder", default="runtime/", help="specifies the path to the folder that will contain the runtime data of the (components in the) workflow")
    arg_parser.add_argument("--workflowResultsFolder", default="results/", help="specifies the path to the folder that will contain the results of the workflow")
    arg_parser.add_argument("--inet_path", default="inet4.5/", help="specifies the base path of inet")
    arg_parser.add_argument("--sims_path", default="sims/", help="specifies the path where the sims will be stored")
    arg_parser.add_argument("--dummy_path", default="sims/", help="specifies the path to the dummy sims")
    arg_parser.add_argument("--platform", default="DAS", help="specifies the platform where it will be running")

    args = arg_parser.parse_args()

    dummy_sim_path = args.dummy_path
    inet_base = args.inet_path
    design_queues = [WorkflowConfig.create_design_point_queue_config("base", 0, "FIFO")]
    sims_path = args.sims_path


    if args.platform == "DAS":
        cluster_config = WorkflowConfig.create_slurm_cluster_config(1, 32, 1, "64GB")
    elif args.platform == "local":
        cluster_config = WorkflowConfig.create_local_cluster_config(1, 8)

    local_config_path = "config"
    local_sim_exec = "run_sim"
    local_results_path = "results"
    local_logs_path = "logs"
    local_out_path = "out"
    uid_scheme = "md5-files"

    workflow_config = WorkflowConfig(sims_path, local_config_path, local_sim_exec, local_results_path, local_logs_path, local_out_path,
                            args.workflowResultsFolder, args.workflowLogsFolder, args.workflowRuntimeFolder, design_queues, uid_scheme, cluster_config)
    workflow_config.write_conf(args.workflowConfigFile)

    config = workflow_config.conf()

    num_sims = 1

    # sim_instances = [create_sim_pdes_comm(config, dummy_sim_path, id, inet_base) for id in range(num_sims)]
    sim_instances = [create_sim_custom_dummy(config, dummy_sim_path, id, inet_base) for id in range(num_sims)]
    # sim_instances = [create_sim_seq_comm(config, dummy_sim_path, id, inet_base) for id in range(num_sims)]

    m = Manager(args.workflowConfigFile, args.workflowLogsFolder)

    m.enqueue_tasks(sim_instances)
    evaluated_sim_instances = m.evaluate_all()
    m.shutdown()


if __name__ == '__main__':
    main()