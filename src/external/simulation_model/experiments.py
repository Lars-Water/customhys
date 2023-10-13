import argparse
import shutil
import os
import json
import random
import time
import pandas as pd

from src.manager import Manager
from src.resourcecontroller import ResourceController
from src.outputhandler import OutputHandler
from src.siminstance import Siminstance
from src.utils.config_creator import OmnetSimConfig, SiminstanceConfig, WorkflowConfig

from create_sims import *

def get_sim_row(path):
    runtime_path = os.path.join(path, "runtime.json")

    with open(runtime_path) as f:
        sim_runtime = json.load(f)

    row = pd.DataFrame(pd.json_normalize(sim_runtime))
    # row = row.set_index("general.UID")
    return row


def get_campaign_sims_data(path):
    runtime_path = os.path.join(path, "runtime")
    sim_paths = [os.path.join(runtime_path, sim_folder) for sim_folder in sorted(os.listdir(runtime_path)) if os.path.isdir(os.path.join(runtime_path, sim_folder))]

    result = pd.DataFrame()

    for sim_path in sim_paths:
        result = pd.concat([result, get_sim_row(sim_path)])

    return result

def get_campaign_data(path, key):
    runtime_path = os.path.join(path, "runtime")

    component_paths = [os.path.join(runtime_path, comp_file) for comp_file in sorted(os.listdir(runtime_path)) if not os.path.isdir(os.path.join(runtime_path, comp_file))]

    result = pd.DataFrame()

    for component_path in component_paths:
        with open(component_path) as f:
            component_data = json.load(f)

        component = os.path.basename(component_path).replace("_runtime.json", "")
        component_flattened = pd.json_normalize(component_data).add_prefix(component + ".")
        component_flattened["workflow"] = key
        # component_flattened = component_flattened.set_index("workflow")

        result = pd.concat([result, component_flattened], axis=1)

    return result

def campaign_run(args, model, num_nodes, num_workers, num_sims, time_stamp, hyper=False):
    if hyper:
        data_path = os.path.join(args.experiments_path, "data", "campaign_hyper_{}".format(model), "n{}_w{}_s{}".format(num_nodes, num_workers, num_sims), time_stamp)
        os.system("python3 experiment_campaign.py --data_path {} --inet_path {} --sims_path {} \
                --dummy_path {} --platform {} --model {} --num_nodes {} --num_workers {} --num_sims {} --hyper".format(
            data_path, args.inet_path, args.sims_path, args.dummy_path, args.platform, model, num_nodes, num_workers, num_sims))
    else:
        data_path = os.path.join(args.experiments_path, "data", "campaign_{}".format(model), "n{}_w{}_s{}".format(num_nodes, num_workers, num_sims), time_stamp)
        os.system("python3 experiment_campaign.py --data_path {} --inet_path {} --sims_path {} \
                --dummy_path {} --platform {} --model {} --num_nodes {} --num_workers {} --num_sims {}".format(
            data_path, args.inet_path, args.sims_path, args.dummy_path, args.platform, model, num_nodes, num_workers, num_sims))

def campaign_run_collect(args, model, num_nodes, num_workers, num_sims, time_stamp):
    results_path = os.path.join(args.experiments_path, "results", "campaign_{}".format(model), "n{}_w{}_s{}".format(num_nodes, num_workers, num_sims), time_stamp)
    data_path = os.path.join(args.experiments_path, "data", "campaign_{}".format(model), "n{}_w{}_s{}".format(num_nodes, num_workers, num_sims), time_stamp)

    os.makedirs(results_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    sims_dataframe = get_campaign_sims_data(data_path)
    csv_file_path = os.path.join(results_path, "sims_runtime.csv")
    sims_dataframe["num_nodes"] = num_nodes
    sims_dataframe["num_workers"] = num_workers
    sims_dataframe["num_sims"] = num_sims
    sims_dataframe["model"] = model
    sims_dataframe["time_stamp"] = time_stamp
    sims_dataframe.to_csv(csv_file_path, index=False)

    workflow_dataframe = get_campaign_data(data_path, "n{}_w{}_s{}".format(num_nodes, num_workers, num_sims))
    workflow_csv_file_path = os.path.join(results_path, "workflow_runtime.csv")
    workflow_dataframe["num_nodes"] = num_nodes
    workflow_dataframe["num_workers"] = num_workers
    workflow_dataframe["num_sims"] = num_sims
    workflow_dataframe["model"] = model
    workflow_dataframe["time_stamp"] = time_stamp

    workflow_dataframe.to_csv(workflow_csv_file_path, index=False)


def campaign_collect(args, model, num_nodes, num_workers, num_sims):
    results_paths = os.path.join(args.experiments_path, "results", "campaign_{}".format(model), "n{}_w{}_s{}".format(num_nodes, num_workers, num_sims))
    campaign_paths = [os.path.join(results_paths, run_folder) for run_folder in sorted(os.listdir(results_paths)) if os.path.isdir(os.path.join(results_paths, run_folder))]

    campaign_sims = pd.DataFrame()
    campaign_workflow = pd.DataFrame()

    for campaign_path in campaign_paths:
        sims_runtime_path = os.path.join(campaign_path, "sims_runtime.csv")
        workflow_runtime_path = os.path.join(campaign_path, "workflow_runtime.csv")

        sims_runtime = pd.read_csv(sims_runtime_path)
        workflow_runtime = pd.read_csv(workflow_runtime_path)

        campaign_sims = pd.concat([campaign_sims, sims_runtime])
        campaign_workflow = pd.concat([campaign_workflow, workflow_runtime])

    campaign_sims_path = os.path.join(results_paths, "sims_runtime.csv")
    campaign_workflow_path = os.path.join(results_paths, "workflow_runtime.csv")

    campaign_sims.to_csv(campaign_sims_path, index=False)
    campaign_workflow.to_csv(campaign_workflow_path, index=False)

def campaign_model_collect(args, model):
    results_paths = os.path.join(args.experiments_path, "results", "campaign_{}".format(model))
    config_paths = [os.path.join(results_paths, config_folder) for config_folder in sorted(os.listdir(results_paths)) if os.path.isdir(os.path.join(results_paths, config_folder))]

    model_sims = pd.DataFrame()
    model_workflow = pd.DataFrame()

    for config_path in config_paths:
        sims_runtime_path = os.path.join(config_path, "sims_runtime.csv")
        workflow_runtime_path = os.path.join(config_path, "workflow_runtime.csv")

        sims_runtime = pd.read_csv(sims_runtime_path)
        workflow_runtime = pd.read_csv(workflow_runtime_path)

        model_sims = pd.concat([model_sims, sims_runtime])
        model_workflow = pd.concat([model_workflow, workflow_runtime])

    model_sims_path = os.path.join(results_paths, "sims_runtime.csv")
    model_workflow_path = os.path.join(results_paths, "workflow_runtime.csv")

    model_sims.to_csv(model_sims_path, index=False)
    model_workflow.to_csv(model_workflow_path, index=False)


def campaign(args, model, num_nodes_list, num_workers_list, num_sims_list, repetitions, hyper=False):
    model_name = model
    if hyper:
        model_name = "hyper_{}".format(model)
    for i, num_sims in enumerate(num_sims_list):
        len_sims = len(num_sims_list)
        for j, num_workers in enumerate(num_workers_list):
            len_workers = len(num_workers_list)
            sim_count_workers = j + len_workers * i
            for k, num_nodes in enumerate(num_nodes_list):
                len_nodes = len(num_nodes_list)
                sim_count_nodes = k + sim_count_workers * len_nodes
                for rep in range(repetitions):
                    run_count = (rep + 1) + repetitions * sim_count_nodes

                    print("Sims: {} from {}".format(num_sims, num_sims_list))
                    print("Workers: {} from {}".format(num_workers, num_workers_list))
                    print("Nodes: {} from {}".format(num_nodes, num_nodes_list))
                    print("Rep: {}/{}".format(rep+1, repetitions))
                    print("Run {} out of {}".format(run_count, len(num_sims_list) * len(num_workers_list) * len(num_nodes_list) * repetitions))

                    time_stamp = time.strftime("%Y%m%d_%H%M%S")

                    campaign_run(args, model, num_nodes, num_workers, num_sims, time_stamp, hyper=hyper)

                    campaign_run_collect(args, model_name, num_nodes, num_workers, num_sims, time_stamp)

                campaign_collect(args, model_name, num_nodes, num_workers, num_sims)

    campaign_model_collect(args, model_name)

def sequential_run(args, model, time_stamp):
    data_path = os.path.join(args.experiments_path, "data", "sequential_{}".format(model), time_stamp)

    if args.platform == "DAS":
        os.system("prun -1 -np 1 python experiment_seq.py --data_path {} --inet_path {} --sims_path {} --dummy_path {} --platform {} --model {}".format(data_path, args.inet_path, args.sims_path, args.dummy_path, args.platform, model))
    else:
        os.system("python3 experiment_seq.py --data_path {} --inet_path {} --sims_path {} --dummy_path {} --platform {} --model {}".format(data_path, args.inet_path, args.sims_path, args.dummy_path, args.platform, model))

def sequential_run_multi(args, model, time_stamp, multi_num):
    data_path = os.path.join(args.experiments_path, "data", "sequential_multi_{}".format(model), time_stamp)

    if args.platform == "DAS":
        os.system("prun -{} -np 1 python experiment_seq.py --data_path {} --inet_path {} --sims_path {} --dummy_path {} --platform {} --model {} --multi".format(multi_num, data_path, args.inet_path, args.sims_path, args.dummy_path, args.platform, model))
    else:
        for i in range(3):
            os.system("python3 experiment_seq.py --data_path {} --inet_path {} --sims_path {} --dummy_path {} --platform {} --model {} --multi".format(data_path, args.inet_path, args.sims_path, args.dummy_path, args.platform, model))


def sequential_run_collect(args, model, time_stamp):
    results_path = os.path.join(args.experiments_path, "results", "sequential_{}".format(model), time_stamp)
    data_path = os.path.join(args.experiments_path, "data", "sequential_{}".format(model), time_stamp)

    os.makedirs(results_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    sims_dataframe = get_campaign_sims_data(data_path)
    csv_file_path = os.path.join(results_path, "sims_runtime.csv")
    sims_dataframe["model"] = model
    sims_dataframe["time_stamp"] = time_stamp
    sims_dataframe.to_csv(csv_file_path, index=False)


def sequential_model_collect(args, model):
    results_paths = os.path.join(args.experiments_path, "results", "sequential_{}".format(model))
    sequential_paths = [os.path.join(results_paths, run_folder) for run_folder in sorted(os.listdir(results_paths)) if os.path.isdir(os.path.join(results_paths, run_folder))]

    sequential_sims = pd.DataFrame()

    for sequential_path in sequential_paths:
        sims_runtime_path = os.path.join(sequential_path, "sims_runtime.csv")

        sims_runtime = pd.read_csv(sims_runtime_path)

        sequential_sims = pd.concat([sequential_sims, sims_runtime])

    sequential_sims_path = os.path.join(results_paths, "sims_runtime.csv")

    sequential_sims.to_csv(sequential_sims_path, index=False)

def sequential_multi_model_collect(args, model):
    results_path_base = os.path.join(args.experiments_path, "results", "sequential_multi_{}".format(model))
    data_path_base = os.path.join(args.experiments_path, "data", "sequential_multi_{}".format(model))

    sequential_paths = [os.path.join(data_path_base, run_folder) for run_folder in sorted(os.listdir(data_path_base)) if os.path.isdir(os.path.join(data_path_base, run_folder))]

    os.makedirs(results_path_base, exist_ok=True)

    csv_file_path = os.path.join(results_path_base, "sims_runtime.csv")

    sequential_sims = pd.DataFrame()

    for sequential_path in sequential_paths:
        time_stamp = os.path.basename(sequential_path)
        ids_paths = [os.path.join(sequential_path, id_folder) for id_folder in sorted(os.listdir(sequential_path)) if os.path.isdir(os.path.join(sequential_path, id_folder))]
        for id_path in ids_paths:
            id = os.path.basename(id_path)
            sims_dataframe = get_campaign_sims_data(id_path)
            sims_dataframe["model"] = model
            sims_dataframe["time_stamp"] = time_stamp
            sims_dataframe["id"] = id

            sequential_sims = pd.concat([sequential_sims, sims_dataframe])

    sequential_sims.to_csv(csv_file_path, index=False)


def sequential_multi(args, models_list, repetitions, multi_nums):
    for i, model in enumerate(models_list):
        for rep in range(repetitions):
            run_count = (rep + 1) + repetitions * i
            print("Sims: {} from {}".format(model, models_list))
            print("Rep: {}/{}".format(rep+1, repetitions))
            print("Run {} out of {}".format(run_count, len(models_list) * repetitions))

            time_stamp = time.strftime("%Y%m%d_%H%M%S")

            sequential_run_multi(args, model, time_stamp, multi_nums[i])

        sequential_multi_model_collect(args, model)

def sequential(args, models_list, repetitions):
    for i, model in enumerate(models_list):
        for rep in range(repetitions):
            run_count = (rep + 1) + repetitions * i
            print("Sims: {} from {}".format(model, models_list))
            print("Rep: {}/{}".format(rep+1, repetitions))
            print("Run {} out of {}".format(run_count, len(models_list) * repetitions))

            time_stamp = time.strftime("%Y%m%d_%H%M%S")

            sequential_run(args, model, time_stamp)
            sequential_run_collect(args, model, time_stamp)

        sequential_model_collect(args, model)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--experiments_path", default="experiments/", help="specifies the base path of the project")
    arg_parser.add_argument("--inet_path", default="inet4.5/", help="specifies the base path of inet")
    arg_parser.add_argument("--sims_path", default="sims/", help="specifies the path where the sims will be stored")
    arg_parser.add_argument("--dummy_path", default="sims/", help="specifies the path to the dummy sims")
    arg_parser.add_argument("--platform", default="DAS", help="specifies the platform where it will be running")

    args = arg_parser.parse_args()

    # num_sims_list = [1]
    # num_nodes_list = [1, 2, 3, 4]
    # num_workers_list = [1]
    # repetitions = 5

    num_sims_list = [1]
    num_nodes_list = [1]
    num_workers_list = [1]
    repetitions = 1

    campaign(args, "seq_comm", num_nodes_list, num_workers_list, num_sims_list, repetitions)
    # campaign(args, "tictoc", num_nodes_list, num_workers_list, num_sims_list, repetitions)

if __name__ == '__main__':
    main()
