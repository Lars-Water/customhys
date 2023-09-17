import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta

def visualise_worker_size(experiments_path, savefig, model, scaling, num_nodes_list, num_workers_list, num_sims_list):
    sims_runtime_data = get_campaign_sims_data(experiments_path, model)
    workflow_runtime_data = get_campaign_workflow_data(experiments_path, model)

    baseline_data = get_sequential_data(experiments_path, model)
    baseline_times = baseline_data["simulation.simulation_time"] + baseline_data["compilation.compile_time"]
    baseline_times_avg = baseline_times.mean()

    baseline_multi_times = get_sequential_times(experiments_path, "multi_{}".format(model))
    baselin_multi_times_avg = baseline_multi_times.mean()

    fig_speedup, (ax_speedup1, ax_speedup2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 0.1]})
    fig_speedup.subplots_adjust(hspace=0.05)


    for num_sims in num_sims_list:
        for num_nodes in num_nodes_list:
            exec_avg = []
            exec_std = []

            sim_avg = []
            sim_std = []

            comp_avg = []
            comp_std = []
            for num_workers in num_workers_list:
                if scaling == "weak":
                    num_sims_scaling = num_sims * num_nodes
                else:
                    num_sims_scaling = num_sims

                campaign_config_data = sims_runtime_data.loc[sims_runtime_data["num_nodes"]==num_nodes]
                campaign_config_data = campaign_config_data.loc[campaign_config_data["num_workers"]==num_workers]
                campaign_config_data = campaign_config_data.loc[campaign_config_data["num_sims"]==num_sims_scaling]

                execution_times = campaign_config_data["general.dask_worker_time"]
                sim_times = campaign_config_data["simulation.simulation_time"]
                comp_times = campaign_config_data["compilation.compile_time"]

                exec_avg += [execution_times.mean()]
                exec_std += [execution_times.std()]

                sim_avg += [execution_times.mean()]
                sim_std += [execution_times.std()]

                comp_avg += [execution_times.mean()]
                comp_std += [execution_times.std()]

            l, _, _ = ax_speedup1.errorbar(num_workers_list, exec_avg, label="Nodes: {}".format(num_nodes), yerr=exec_std)
            # l, _, _ = plt.errorbar(num_workers_list, sim_avg, label="Nodes: {}, Sims: {}".format(num_nodes, num_sims), color=l.get_color(), yerr=sim_std, linestyle=':')
            # plt.errorbar(num_workers_list, comp_avg, label="Nodes: {}, Sims: {}".format(num_nodes, num_sims), color=l.get_color(), yerr=comp_std, linestyle='dashdot')
    ax_speedup1.set_ylim(115, 155)
    ax_speedup2.set_ylim(0, 5)

    ax_speedup1.set_xticks(num_workers_list)
    ax_speedup2.set_xticks(num_workers_list)

    ax_speedup1.spines['bottom'].set_visible(False)
    ax_speedup2.spines['top'].set_visible(False)

    ax_speedup1.xaxis.tick_top()
    # ax_speedup1.xaxis.set_tick_params(labeltop=False)
    ax_speedup1.xaxis.set_tick_params(top=False)
    ax_speedup1.tick_params(labeltop=False)
    ax_speedup2.xaxis.tick_bottom()

    ax_speedup1.set_yticks(ax_speedup1.get_yticks()[1:])
    ax_speedup2.set_yticks(ax_speedup2.get_yticks()[:-1])

    d = 0.5

    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=12,
        linestyle='none',
        color='k',
        mec='k',
        mew=1,
        clip_on=False
    )
    ax_speedup1.plot([0, 1], [0, 0], transform=ax_speedup1.transAxes, **kwargs)
    ax_speedup2.plot([0, 1], [1, 1], transform=ax_speedup2.transAxes, **kwargs)

    ax_speedup1.axhline(baseline_times_avg, label="Sequential baseline", linestyle=':', color="gray")
    ax_speedup1.axhline(baselin_multi_times_avg, label="Sequential Multi baseline", linestyle='--', color="gray")


    ax_speedup1.set_title("Execution time for increasing worker sizes")
    ax_speedup2.set_xlabel("Worker Size")
    ax_speedup1.set_ylabel("Execution time (s)")
    # plt.ylim(ymin=120)
    # plt.xticks(ticks=num_workers_list)
    ax_speedup1.grid()
    ax_speedup2.grid()
    ax_speedup1.legend(loc='lower right')
    # plt.legend()
    out_path_base = os.path.join(experiments_path, "figures" , "campaign_{}".format(model))
    os.makedirs(out_path_base, exist_ok=True)
    fig_speedup.savefig(os.path.join(out_path_base, "{}.png".format(savefig)))
    fig_speedup.savefig(os.path.join(out_path_base, "{}.pdf".format(savefig)))
    # plt.show()

def visualise_gantt(experiments_path, savefig, model, num_nodes, num_workers, num_sims, time_stamp=None):
    if not time_stamp:
        base_path = workflow_runtime_path = os.path.join(experiments_path, "results", "campaign_{}".format(model), "n{}_w{}_s{}".format(num_nodes, num_workers, num_sims))
        iter_paths = [os.path.join(base_path, iter_folder) for iter_folder in sorted(os.listdir(base_path)) if os.path.isdir(os.path.join(base_path, iter_folder))]
        time_stamp = os.path.basename(iter_paths[0])

    sims_runtime_data = get_campaign_config_sims_data(experiments_path, model, num_nodes, num_workers, num_sims, time_stamp)
    workflow_runtime_data = get_campaign_config_workflow_data(experiments_path, model, num_nodes, num_workers, num_sims, time_stamp)

    workers = sims_runtime_data["general.dask_worker"].unique()

    for i, worker in enumerate(workers):
        sims_runtime_data["general.dask_worker"] = sims_runtime_data["general.dask_worker"].replace({worker : "Worker {}".format(i)})

    sims_runtime_data = sims_runtime_data.sort_values(by="general.dask_worker_start", ascending=False)
    earliest_time = sims_runtime_data["general.dask_worker_start"].min()
    sims_runtime_data["general.dask_worker_start"] = sims_runtime_data["general.dask_worker_start"] - earliest_time
    sims_runtime_data["general.dask_worker_end"] = sims_runtime_data["general.dask_worker_end"] - earliest_time

    workers = sims_runtime_data["general.dask_worker"].values
    uids = sims_runtime_data["general.UID"].values
    task_start = sims_runtime_data["general.dask_worker_start"].values
    task_end = sims_runtime_data["general.dask_worker_end"].values

    data = [dict(Worker=w, Task=u, Start=datetime.utcfromtimestamp(s), Finish=datetime.utcfromtimestamp(f)) for w, u, s, f in zip(workers, uids, task_start, task_end)]
    df = pd.DataFrame(data)

    # Without group_tasks=True, There would be two separate "Job A" labels
    # fig = ff.create_gantt(data, group_tasks=True)

    ## plot the non-duplicate rows
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Worker", color="Task")
    fig.update_layout(barmode='group', showlegend=False, xaxis_title="Time (s)")
    fig.update_xaxes(tickformat="%s")

    # ## plot the duplicate rows using rectangular shapes
    # for row in df.loc[df['Worker'].duplicated()].itertuples():
    #     y_val = np.where(df.Worker.unique()==row[1])[0][0]
    #     print(f"found {row[1]} at index {y_val}")
    #     fig.add_shape(type="rect",
    #             xref="x", yref="y",
    #             x0=row[3], x1=row[4],
    #             y0=y_val, y1=y_val-0.4,
    #             # line_width=0,
    #             # fillcolor="salmon",
    # )
    out_path_base = os.path.join(experiments_path, "figures" , "campaign_{}".format(model))
    os.makedirs(out_path_base, exist_ok=True)
    fig.write_image(os.path.join(out_path_base, "{}.png".format(savefig)), scale=2)
    fig.write_image(os.path.join(out_path_base, "{}.pdf".format(savefig)), scale=2)

def get_campaign_config_sims_data(experiments_path, model, num_nodes, num_workers, num_sims, time_stamp):
    workflow_runtime_path = os.path.join(experiments_path, "results", "campaign_{}".format(model), "n{}_w{}_s{}".format(num_nodes, num_workers, num_sims), time_stamp, "sims_runtime.csv")
    workflow_runtime_data = pd.read_csv(workflow_runtime_path)
    return workflow_runtime_data

def get_campaign_sims_data(experiments_path, model):
    sim_runtime_path = os.path.join(experiments_path, "results", "campaign_{}".format(model), "sims_runtime.csv")
    sim_runtime_data = pd.read_csv(sim_runtime_path)
    return sim_runtime_data

def get_campaign_config_workflow_data(experiments_path, model, num_nodes, num_workers, num_sims, time_stamp):
    workflow_runtime_path = os.path.join(experiments_path, "results", "campaign_{}".format(model), "n{}_w{}_s{}".format(num_nodes, num_workers, num_sims), time_stamp, "workflow_runtime.csv")
    workflow_runtime_data = pd.read_csv(workflow_runtime_path)
    return workflow_runtime_data

def get_campaign_workflow_data(experiments_path, model):
    workflow_runtime_path = os.path.join(experiments_path, "results", "campaign_{}".format(model), "workflow_runtime.csv")
    workflow_runtime_data = pd.read_csv(workflow_runtime_path)
    return workflow_runtime_data

def get_sequential_data(experiments_path, model):
    sequential_runtime_path = os.path.join(experiments_path, "results", "sequential_{}".format(model), "sims_runtime.csv")
    sequential_runtime_data = pd.read_csv(sequential_runtime_path)
    return sequential_runtime_data

def get_baseline_times(baseline_data):
    return baseline_data["simulation.simulation_time"] + baseline_data["compilation.compile_time"]

def get_sequential_times(experiments_path, model):
    baseline_data = get_sequential_data(experiments_path, model)
    return get_baseline_times(baseline_data)

def get_config_data(data, num_nodes, num_workers, num_sims):
    result = data.loc[data["num_nodes"]==num_nodes]
    result = result.loc[result["num_workers"]==num_workers]
    result = result.loc[result["num_sims"]==num_sims]
    return result


def get_num_sims_scaling(num_nodes, num_sims, scaling):
    if scaling == "strong":
        num_sims_scaling = num_sims
    else:
        num_sims_scaling = num_sims * num_nodes

    return num_sims_scaling

def get_campaign_speedups(experiments_path, model, scaling, num_sims, hyper=False):
    if hyper:
        campaign_data = get_campaign_workflow_data(experiments_path, "hyper_{}".format(model))
    else:
        campaign_data = get_campaign_workflow_data(experiments_path, model)

    baseline_times = get_sequential_times(experiments_path, model)
    baseline_multi_times = get_sequential_times(experiments_path, "multi_{}".format(model))

    environment_time_key = "manager.general.environment_time"

    campaign_data["num_sims_scaling"] = campaign_data.apply(lambda x: get_num_sims_scaling(x["num_nodes"], num_sims, scaling), axis=1)

    campaign_data["seq_total"] = baseline_times.mean() * campaign_data["num_sims_scaling"]
    campaign_data["seq_multi_total"] = baseline_multi_times.mean() * campaign_data["num_sims_scaling"]

    campaign_data["speedup_seq"] = campaign_data["seq_total"] / campaign_data[environment_time_key]
    campaign_data["speedup_seq_multi"] = campaign_data["seq_multi_total"] / campaign_data[environment_time_key]

    return campaign_data

def get_campaign_relative_speedups(experiments_path, baselines, model1, hyper1, model2, hyper2, scaling, num_nodes, num_sims, num_workers, num_cores):
    campaign_data_model1 = get_campaign_efficiencies(experiments_path, model1, scaling, num_sims, num_cores, hyper=hyper1)
    campaign_data_model2 = get_campaign_efficiencies(experiments_path, model2, scaling, num_sims, num_cores, hyper=hyper2)


    campaign_data1 = campaign_data_model1.loc[campaign_data_model1["num_nodes"] == num_nodes]
    campaign_data1 = campaign_data1.loc[campaign_data1["num_sims"] == num_sims]
    campaign_data1 = campaign_data1.loc[campaign_data1["num_workers"] == num_workers]

    campaign_data2 = campaign_data_model2.loc[campaign_data_model2["num_nodes"] == num_nodes]
    campaign_data2 = campaign_data2.loc[campaign_data2["num_sims"] == num_sims]
    campaign_data2 = campaign_data2.loc[campaign_data2["num_workers"] == num_workers]

    # if (num_nodes == 4):
        # print(scaling, num_nodes, num_sims)
        # print(campaign_data1[["speedup_{}".format(baseline) for baseline in baselines]])
        # print(campaign_data2[["speedup_{}".format(baseline) for baseline in baselines]])


    # baselines = ["seq", "seq_multi"]
    rspeedups_model1 = []
    # print(campaign_data1.columns)

    for baseline in baselines:
        # print(baseline)
        # print(campaign_data1["speedup_{}".format(baseline)].mean())
        rspeedup = campaign_data2["speedup_{}".format(baseline)].mean() / campaign_data1["speedup_{}".format(baseline)].mean() - 1
        # if num_nodes == 4:
            # print(baseline, rspeedup)
        rspeedups_model1 += [rspeedup]

    return rspeedups_model1

def get_pdes_relative_speedups(experiments_path, baselines, model1, model2, scaling, num_nodes, num_sims, num_workers, num_cores):
    campaign_data_model1 = get_pdes_efficiencies(experiments_path, model1, scaling, num_sims, num_cores)
    campaign_data_model2 = get_pdes_efficiencies(experiments_path, model2, scaling, num_sims, num_cores)


    campaign_data1 = campaign_data_model1.loc[campaign_data_model1["num_nodes"] == num_nodes]
    campaign_data1 = campaign_data1.loc[campaign_data1["num_sims"] == num_sims]
    campaign_data1 = campaign_data1.loc[campaign_data1["num_workers"] == num_workers]

    campaign_data2 = campaign_data_model2.loc[campaign_data_model2["num_nodes"] == num_nodes]
    campaign_data2 = campaign_data2.loc[campaign_data2["num_sims"] == num_sims]
    campaign_data2 = campaign_data2.loc[campaign_data2["num_workers"] == num_workers]

    # if (num_nodes == 4):
        # print(scaling, num_nodes, num_sims)
        # print(campaign_data1[["speedup_{}".format(baseline) for baseline in baselines]])
        # print(campaign_data2[["speedup_{}".format(baseline) for baseline in baselines]])


    # baselines = ["seq", "seq_multi"]
    rspeedups_model1 = []
    refficiencies_model1 = []
    # print(campaign_data1.columns)

    for baseline in baselines:
        # print(baseline)
        # print(campaign_data1["speedup_{}".format(baseline)].mean())
        rspeedup = campaign_data2["speedup_{}".format(baseline)].mean() / campaign_data1["speedup_{}".format(baseline)].mean() - 1
        refficiency = campaign_data2["efficiency_{}".format(baseline)].mean() / campaign_data1["efficiency_{}".format(baseline)].mean() - 1
        # if num_nodes == 4:
            # print(baseline, rspeedup)
        rspeedups_model1 += [rspeedup]
        refficiencies_model1 += [rspeedup]

    return rspeedups_model1, refficiencies_model1


def get_campaign_efficiencies(experiments_path, model, scaling, num_sims, num_cores, hyper=False):
    campaign_data = get_campaign_speedups(experiments_path, model, scaling, num_sims, hyper=hyper)

    if hyper:
        campaign_data["efficiency_seq"] = campaign_data["speedup_seq"] / (campaign_data["num_nodes"] * num_cores)
        campaign_data["efficiency_seq_multi"] = campaign_data["speedup_seq_multi"] / (campaign_data["num_nodes"] * num_cores)
    else:
        campaign_data["efficiency_seq"] = campaign_data["speedup_seq"] / (campaign_data["num_nodes"] * num_cores)
        campaign_data["efficiency_seq_multi"] = campaign_data["speedup_seq_multi"] / (campaign_data["num_nodes"] * num_cores)

    return campaign_data

def get_pdes_speedups(experiments_path, basemodel, scaling, num_sims):
    campaign_data = get_campaign_workflow_data(experiments_path, "pdes_{}".format(basemodel))

    environment_time_key = "manager.general.environment_time"
    num_lps = 8

    baseline_times = get_sequential_times(experiments_path, "seq_{}".format(basemodel))
    baseline_multi_times = get_sequential_times(experiments_path, "multi_seq_{}".format(basemodel))
    baseline_campaign_data = get_campaign_workflow_data(experiments_path, "seq_{}".format(basemodel))
    baseline_pdes_times = get_sequential_times(experiments_path, "pdes_{}".format(basemodel))
    baseline_pdes_multi_times = get_sequential_times(experiments_path, "multi_pdes_{}".format(basemodel))

    campaign_data["num_sims_scaling"] = campaign_data.apply(lambda x: get_num_sims_scaling(x["num_nodes"], num_sims, scaling), axis=1)

    campaign_data["seq_total"] = baseline_times.mean() * campaign_data["num_sims_scaling"]
    campaign_data["seq_multi_total"] = baseline_multi_times.mean() * campaign_data["num_sims_scaling"]
    campaign_data["seq_campaign_total"] = campaign_data.apply(lambda x: get_config_data(baseline_campaign_data, x["num_nodes"], x["num_workers"], x["num_sims_scaling"])[environment_time_key].mean(), axis=1)
    campaign_data["pdes_total"] = baseline_pdes_times.mean() * campaign_data["num_sims_scaling"]
    # print(baseline_pdes_times.mean())
    # print(scaling)
    # print(campaign_data["num_nodes"][80])
    # print(campaign_data[environment_time_key][80])
    # print(campaign_data["pdes_total"][80])
    # print(campaign_data["num_sims"][80])
    # print(campaign_data["num_sims_scaling"][80])
    campaign_data["pdes_multi_total"] = baseline_pdes_multi_times.mean() * campaign_data["num_sims_scaling"]

    campaign_data["speedup_seq"] = campaign_data["seq_total"] / campaign_data[environment_time_key]
    campaign_data["speedup_seq_multi"] = campaign_data["seq_multi_total"] / campaign_data[environment_time_key]
    campaign_data["speedup_campaign"] = campaign_data["seq_campaign_total"] / campaign_data[environment_time_key]
    campaign_data["speedup_pdes"] = campaign_data["pdes_total"] / campaign_data[environment_time_key]
    # print(campaign_data["speedup_pdes"][80])
    campaign_data["speedup_pdes_multi"] = campaign_data["pdes_multi_total"] / campaign_data[environment_time_key]

    return campaign_data


def get_pdes_efficiencies(experiments_path, basemodel, scaling, num_sims, num_cores):
    campaign_data = get_pdes_speedups(experiments_path, basemodel, scaling, num_sims)

    # TODO: make the number of lps (8) configurable
    num_lps = 8

    campaign_data["efficiency_seq"] = campaign_data["speedup_seq"] / (campaign_data["num_nodes"] * num_cores)
    campaign_data["efficiency_seq_multi"] = campaign_data["speedup_seq_multi"] / (campaign_data["num_nodes"] * num_cores)
    campaign_data["efficiency_campaign"] = campaign_data["speedup_campaign"]
    campaign_data["efficiency_pdes"] = campaign_data["speedup_pdes"] / (campaign_data["num_nodes"] * num_cores / num_lps)
    # print(campaign_data["efficiency_pdes"][80])
    campaign_data["efficiency_pdes_multi"] = campaign_data["speedup_pdes_multi"] / (campaign_data["num_nodes"] * num_cores / num_lps)

    return campaign_data


def get_key_data(data, key, num_nodes_list, num_workers, num_sims, scaling):
    # print(get_num_sims_scaling(3, num_sims, scaling))
    return [get_config_data(data, num_nodes, num_workers, get_num_sims_scaling(num_nodes, num_sims, scaling))[key].tolist() for num_nodes in num_nodes_list]

# FROM: https://stackoverflow.com/a/60270421
def plot_bars(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            local_values = np.array(y)
            bar = ax.bar(x + 1 + x_offset, local_values.mean(), width=bar_width * single_width, color=colors[i % len(colors)], yerr=local_values.std(), zorder=3)

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())

# FROM: https://stackoverflow.com/a/60270421
def plot_bars_speedup_relative(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + 1 + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)], zorder=3)

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())

def plot_speedup_r(experiments_path, model1, model2, speedupr_data, run_type, savefig, scaling, num_nodes_list, num_sims, num_cores, cap_efficiency=True):
    fig_speedup, ax_speedup = plt.subplots()
    # print(speedupr_data)

    for i, (name, values) in enumerate(speedupr_data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            local_values = np.array(y)
            bar = ax.bar(x + 1 + x_offset, local_values.mean(), width=bar_width * single_width, color=colors[i % len(colors)], yerr=local_values.std(), zorder=3)

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    ax_speedup.set_title("Relative speedup for an increasing number of nodes")
    ax_speedup.set_xlabel("Number of Nodes")
    ax_speedup.set_ylabel("Relative speedup")
    ax_speedup.set_ylim(ymin=-1.1, ymax=1.1)
    ax_speedup.set_xticks(ticks=num_nodes_list)
    ax_speedup.grid(zorder=-1.0)


    out_path_base = os.path.join(experiments_path, "figures" , "{}_{}".format(run_type, model1))
    os.makedirs(out_path_base, exist_ok=True)

    fig_speedup.savefig(os.path.join(out_path_base, "speedupr_{}_{}_{}.png".format(model1, model2, savefig)))
    fig_speedup.savefig(os.path.join(out_path_base, "speedupr_{}_{}_{}.pdf".format(model1, model2, savefig)))

def plot_speedup_r_lines(experiments_path, model1, model2, nodes_data, run_type, savefig, scaling, num_nodes_list, num_sims_list, num_cores, cap_efficiency=True):
    # print(nodes_data)
    for i, (baseline, values) in enumerate(nodes_data.items()):
        fig_speedup, ax_speedup = plt.subplots()
        # Draw a bar for every value of that type
        for x, node_count_values in enumerate(values):
            ax_speedup.errorbar(num_sims_list, node_count_values, zorder=3, label="Nodes: {}".format(num_nodes_list[x]))

        ax_speedup.set_title("Relative speedup for an increasing number of evaluations")
        ax_speedup.set_xlabel("Number of Evaluations")
        ax_speedup.set_ylabel("Relative speedup")
        ax_speedup.axhline(0, color='black')
        ax_speedup.set_ylim(ymin=-1.1, ymax=1.1)
        ax_speedup.set_xticks(ticks=num_sims_list)
        ax_speedup.grid(zorder=-1.0)
        ax_speedup.legend()


        out_path_base = os.path.join(experiments_path, "figures" , "{}_{}".format(run_type, model1))
        os.makedirs(out_path_base, exist_ok=True)

        fig_speedup.savefig(os.path.join(out_path_base, "speedupr_lines_{}_{}_{}_{}.png".format(model1, model2, baseline, savefig)))
        fig_speedup.savefig(os.path.join(out_path_base, "speedupr_lines_{}_{}_{}_{}.pdf".format(model1, model2, baseline, savefig)))



def plot_efficiency_r(experiments_path, model1, model2, efficiencyr_data, run_type, savefig, scaling, num_nodes_list, num_sims, num_cores, cap_efficiency=True):
    fig_speedup, ax_speedup = plt.subplots()
    # print(speedupr_data)
    plot_bars(ax_speedup, efficiencyr_data)

    ax_speedup.set_title("Relative efficiency for an increasing number of nodes")
    ax_speedup.set_xlabel("Number of Nodes")
    ax_speedup.set_ylabel("Relative efficiency")
    ax_speedup.set_ylim(ymin=-1.1, ymax=1.1)
    ax_speedup.set_xticks(ticks=num_nodes_list)
    ax_speedup.grid(zorder=-1.0)


    out_path_base = os.path.join(experiments_path, "figures" , "{}_{}".format(run_type, model1))
    os.makedirs(out_path_base, exist_ok=True)

    fig_speedup.savefig(os.path.join(out_path_base, "efficiencyr_{}_{}_{}.png".format(model1, model2, savefig)))
    fig_speedup.savefig(os.path.join(out_path_base, "efficiencyr_{}_{}_{}.pdf".format(model1, model2, savefig)))



def plot_speedup_efficiency(experiments_path, speedup_data, efficiency_data, run_type, savefig, model, scaling, num_nodes_list, num_sims, num_cores, cap_efficiency=True):
    fig_speedup, ax_speedup = plt.subplots()
    fig_efficiency, ax_efficiency = plt.subplots()

    plot_bars(ax_speedup, speedup_data)
    plot_bars(ax_efficiency, efficiency_data)

    ax_speedup.set_title("Speedup for an increasing number of nodes")
    ax_speedup.set_xlabel("Number of Nodes")
    ax_speedup.set_ylabel("Speedup")
    ax_speedup.set_ylim(ymin=0, ymax=(num_cores * max(num_nodes_list)))
    ax_speedup.set_xticks(ticks=num_nodes_list)
    ax_speedup.grid(zorder=-1.0)

    ax_efficiency.set_title("Efficiency for an increasing number of nodes")
    ax_efficiency.set_xlabel("Number of Nodes")
    ax_efficiency.set_ylabel("Efficiency")
    if cap_efficiency:
        ax_efficiency.set_ylim(ymin=0, ymax=1.0)
    ax_efficiency.set_xticks(ticks=num_nodes_list)
    ax_efficiency.grid(zorder=-1.0)

    out_path_base = os.path.join(experiments_path, "figures" , "{}_{}".format(run_type, model))
    os.makedirs(out_path_base, exist_ok=True)

    fig_speedup.savefig(os.path.join(out_path_base, "speedup_{}.png".format(savefig)))
    fig_speedup.savefig(os.path.join(out_path_base, "speedup_{}.pdf".format(savefig)))

    fig_efficiency.savefig(os.path.join(out_path_base, "efficiency_{}.png".format(savefig)))
    fig_efficiency.savefig(os.path.join(out_path_base, "efficiency_{}.pdf".format(savefig)))

def campaign_visualise_speedup_efficiency(experiments_path, savefig, model, scaling, num_nodes_list, num_workers, num_sims, num_cores, hyper=False):
    campaign_data = get_campaign_efficiencies(experiments_path, model, scaling, num_sims, num_cores, hyper=hyper)

    fig_speedup, ax_speedup = plt.subplots()
    fig_efficiency, ax_efficiency = plt.subplots()

    speedup_seq = get_key_data(campaign_data, "speedup_seq", num_nodes_list, num_workers, num_sims, scaling)
    speedup_seq_multi = get_key_data(campaign_data, "speedup_seq_multi", num_nodes_list, num_workers, num_sims, scaling)

    efficiency_seq = get_key_data(campaign_data, "efficiency_seq", num_nodes_list, num_workers, num_sims, scaling)
    efficiency_seq_multi = get_key_data(campaign_data, "efficiency_seq_multi", num_nodes_list, num_workers, num_sims, scaling)

    speedup_data = {"Sequential": speedup_seq, "Sequential Multi": speedup_seq_multi}
    efficiency_data = {"Sequential": efficiency_seq, "Sequential Multi": efficiency_seq_multi}

    run_type = "campaign"
    plot_speedup_efficiency(experiments_path, speedup_data, efficiency_data, run_type, savefig, model, scaling, num_nodes_list, num_sims, num_cores)


def pdes_visualise_relative_speedup(experiments_path, savefig, scaling, num_nodes_list, num_workers, num_sims, num_cores):
    baselines = ["seq", "seq_multi", "campaign", "pdes", "pdes_multi"]
    labels = ["Sequential", "Sequential multi", "Sequential campaign", "Sequential PDES", "Sequential PDES multi"]
    models = ["comm", "comp", "sub"]
    for model1 in models:
        for model2 in models:
            print(model1, model2)
            speedupr_data = []
            efficiencyr_data = []
            for num_nodes in num_nodes_list:
                speedupr, efficiencyr = get_pdes_relative_speedups(experiments_path, baselines, model1, model2, scaling, num_nodes, num_sims, num_workers, num_cores)
                speedupr_data += [speedupr]
                efficiencyr_data += [efficiencyr]
            speedupr_data = list(map(list, zip(*speedupr_data)))
            efficiencyr_data = list(map(list, zip(*efficiencyr_data)))
            speedupr_data = {label: data for label, data in zip(labels, speedupr_data)}
            efficiencyr_data = {label: data for label, data in zip(labels, efficiencyr_data)}
            run_type = "campaign"
            plot_speedup_r(experiments_path, "pdes_{}".format(model1), "pdes_{}".format(model2), speedupr_data, run_type, savefig, scaling, num_nodes_list, num_sims, num_cores)
            plot_efficiency_r(experiments_path, "pdes_{}".format(model1), "pdes_{}".format(model2), efficiencyr_data, run_type, savefig, scaling, num_nodes_list, num_sims, num_cores)


def campaign_visualise_relative_speedup(experiments_path, savefig, scaling, baselines, labels, models, hypers, num_nodes_list, num_workers, num_sims, num_cores):
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            print(model1, model2)
            speedupr_data = []
            hyper1 = hypers[i]
            hyper2 = hypers[j]
            for num_nodes in num_nodes_list:
                speedupr_data += [get_campaign_relative_speedups(experiments_path, baselines, model1, hyper1, model2, hyper2, scaling, num_nodes, num_sims, num_workers, num_cores)]
            speedupr_data = list(map(list, zip(*speedupr_data)))
            fig_speedup, ax_speedup = plt.subplots()
            speedupr_data = {label: data for label, data in zip(labels, speedupr_data)}

            run_type = "campaign"
            if hyper1:
                model1_name = "hyper_{}".format(model1)
            else:
                model1_name = model1

            if hyper2:
                model2_name = "hyper_{}".format(model2)
            else:
                model2_name = model2

            plot_speedup_r(experiments_path, model1_name, model2_name, speedupr_data, run_type, savefig, scaling, num_nodes_list, num_sims, num_cores)


def pdes_visualise_relative_speedup_lines(experiments_path, savefig, scaling, num_nodes_list, num_workers, num_sims_list, num_cores):
    baselines = ["seq", "seq_multi", "campaign", "pdes", "pdes_multi"]
    labels = ["Sequential", "Sequential multi", "Sequential campaign", "Sequential PDES", "Sequential PDES multi"]
    models = ["comm", "comp", "sub"]
    for model1 in models:
        for model2 in models:
            print(model1, model2)
            # efficiencyr_data = []
            nodes_data = []
            for num_nodes in num_nodes_list:
                speedupr_data = []
                for num_sims in num_sims_list:
                    speedupr, efficiencyr = get_pdes_relative_speedups(experiments_path, baselines, model1, model2, scaling, num_nodes, num_sims, num_workers, num_cores)
                    speedupr_data += [speedupr]
                speedupr_data = list(map(list, zip(*speedupr_data)))
                nodes_data += [speedupr_data]
            nodes_data = list(map(list, zip(*nodes_data)))
                # efficiencyr_data += [efficiencyr]
            # efficiencyr_data = list(map(list, zip(*efficiencyr_data)))
            nodes_data = {label: data for label, data in zip(labels, nodes_data)}
            # efficiencyr_data = {label: data for label, data in zip(labels, efficiencyr_data)}
            run_type = "campaign"
            plot_speedup_r_lines(experiments_path, "pdes_{}".format(model1), "pdes_{}".format(model2), nodes_data, run_type, savefig, scaling, num_nodes_list, num_sims_list, num_cores)
            # plot_efficiency_r(experiments_path, "pdes_{}".format(model1), "pdes_{}".format(model2), efficiencyr_data, run_type, savefig, scaling, num_nodes_list, num_sims, num_cores)



def campaign_visualise_relative_speedup_line(experiments_path, savefig, scaling, baselines, labels, models, hypers, num_nodes_list, num_workers, num_sims_list, num_cores):
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            print(model1, model2)
            hyper1 = hypers[i]
            hyper2 = hypers[j]
            nodes_data = []
            for num_nodes in num_nodes_list:
                speedupr_data = []
                for num_sims in num_sims_list:
                    speedupr_data += [get_campaign_relative_speedups(experiments_path, baselines, model1, hyper1, model2, hyper2, scaling, num_nodes, num_sims, num_workers, num_cores)]
                speedupr_data = list(map(list, zip(*speedupr_data)))
                nodes_data += [speedupr_data]
            nodes_data = list(map(list, zip(*nodes_data)))
            # print(nodes_data)

            nodes_data = {label: data for label, data in zip(labels, nodes_data)}

            run_type = "campaign"
            if hyper1:
                model1_name = "hyper_{}".format(model1)
            else:
                model1_name = model1

            if hyper2:
                model2_name = "hyper_{}".format(model2)
            else:
                model2_name = model2

            plot_speedup_r_lines(experiments_path, model1_name, model2_name, nodes_data, run_type, savefig, scaling, num_nodes_list, num_sims_list, num_cores)



def pdes_visualise_speedup_efficiency(experiments_path, savefig, base_model, scaling, num_nodes_list, num_workers, num_sims, num_cores):
    model = "pdes_{}".format(base_model)
    campaign_data = get_pdes_efficiencies(experiments_path, base_model, scaling, num_sims, num_cores)

    fig_speedup, ax_speedup = plt.subplots()
    fig_efficiency, ax_efficiency = plt.subplots()

    labels = ["Sequential", "Sequential Multi", "Sequential Campaign", "Sequential PDES", "Sequential PDES Multi"]

    speedup_keys = ["speedup_seq", "speedup_seq_multi", "speedup_campaign", "speedup_pdes", "speedup_pdes_multi"]
    speedup_data = [get_key_data(campaign_data, key, num_nodes_list, num_workers, num_sims, scaling) for key in speedup_keys]

    efficiency_keys = ["efficiency_seq", "efficiency_seq_multi", "efficiency_campaign", "efficiency_pdes", "efficiency_pdes_multi"]
    efficiency_data = [get_key_data(campaign_data, key, num_nodes_list, num_workers, num_sims, scaling) for key in efficiency_keys]

    speedup_data = {label: data for label, data in zip(labels, speedup_data)}
    efficiency_data = {label: data for label, data in zip(labels, efficiency_data)}
    # print(efficiency_data)

    run_type = "campaign"
    plot_speedup_efficiency(experiments_path, speedup_data, efficiency_data, run_type, savefig, model, scaling, num_nodes_list, num_sims, num_cores, cap_efficiency=False)


def visualise_sims_speedup(experiments_path, savefig, model, scaling, num_nodes_list, num_workers, num_sims_list, num_cores, baseline="sequential", hyper=False):
    if hyper:
        sims_runtime_data = get_campaign_sims_data(experiments_path, "hyper_{}".format(model))
        workflow_runtime_data = get_campaign_workflow_data(experiments_path, "hyper_{}".format(model))
    else:
        sims_runtime_data = get_campaign_sims_data(experiments_path, model)
        workflow_runtime_data = get_campaign_workflow_data(experiments_path, model)

    plt.figure()

    if baseline == "sequential":
        baseline_data = get_sequential_data(experiments_path, model)
        baseline_times = baseline_data["simulation.simulation_time"] + baseline_data["compilation.compile_time"]
        baseline_times_avg = baseline_times.mean()
    elif baseline == "sequential_multi":
        baseline_multi_times = get_sequential_times(experiments_path, "multi_{}".format(model))
        baseline_times_avg = baseline_multi_times.mean()
    # print(baseline_times_avg)

    for num_nodes in num_nodes_list:
        avg_speedups = []
        std_speedups = []

        for num_sims in num_sims_list:
            campaign_config_data = workflow_runtime_data.loc[workflow_runtime_data["num_nodes"]==num_nodes]
            campaign_config_data = campaign_config_data.loc[campaign_config_data["num_workers"]==num_workers]

            if scaling == "strong":
                num_sims_scaling = num_sims
                sequential_total = baseline_times_avg * num_sims_scaling
            else:
                num_sims_scaling = num_sims * num_nodes
                sequential_total = baseline_times_avg * num_sims_scaling

            campaign_config_data = campaign_config_data.loc[campaign_config_data["num_sims"]==num_sims_scaling]

            campaign_config_data["seq_total"] = sequential_total

            speedup = campaign_config_data["seq_total"] / campaign_config_data["manager.general.environment_time"]
            efficiency = speedup / (num_nodes * num_cores)
            avg_speedup = speedup.mean()
            std_speedup = speedup.std()

            avg_speedups += [avg_speedup]
            std_speedups += [std_speedup]

        plt.errorbar(num_sims_list, avg_speedups, yerr=std_speedups, zorder=3, label="Nodes: {}".format(num_nodes))

    plt.title("Speedup for an increasing number of evaluations")
    plt.xlabel("Number of Evaluations")
    plt.ylabel("Speedup")
    plt.ylim(ymin=0, ymax=128)
    plt.xticks(ticks=num_sims_list)
    plt.grid(zorder=-1.0)
    plt.legend()

    out_path_base = os.path.join(experiments_path, "figures" , "campaign_{}".format(model))
    os.makedirs(out_path_base, exist_ok=True)
    plt.savefig(os.path.join(out_path_base, "{}.png".format(savefig)))
    plt.savefig(os.path.join(out_path_base, "{}.pdf".format(savefig)))
    # plt.show()


def visualise_sims_efficiency(experiments_path, savefig, model, scaling, num_nodes_list, num_workers, num_sims_list, num_cores, baseline="sequential", hyper=False):
    if hyper:
        sims_runtime_data = get_campaign_sims_data(experiments_path, "hyper_{}".format(model))
        workflow_runtime_data = get_campaign_workflow_data(experiments_path, "hyper_{}".format(model))
    else:
        sims_runtime_data = get_campaign_sims_data(experiments_path, model)
        workflow_runtime_data = get_campaign_workflow_data(experiments_path, model)

    plt.figure()

    if baseline == "sequential":
        baseline_data = get_sequential_data(experiments_path, model)
        baseline_times = baseline_data["simulation.simulation_time"] + baseline_data["compilation.compile_time"]
        baseline_times_avg = baseline_times.mean()
    elif baseline == "sequential_multi":
        baseline_multi_times = get_sequential_times(experiments_path, "multi_{}".format(model))
        baseline_times_avg = baseline_multi_times.mean()
    # print(baseline_times_avg)

    for num_nodes in num_nodes_list:
        avg_efficiencies = []
        std_efficiencies = []

        for num_sims in num_sims_list:
            campaign_config_data = workflow_runtime_data.loc[workflow_runtime_data["num_nodes"]==num_nodes]
            campaign_config_data = campaign_config_data.loc[campaign_config_data["num_workers"]==num_workers]

            if scaling == "strong":
                num_sims_scaling = num_sims
                sequential_total = baseline_times_avg * num_sims_scaling
            else:
                num_sims_scaling = num_sims * num_nodes
                sequential_total = baseline_times_avg * num_sims_scaling

            campaign_config_data = campaign_config_data.loc[campaign_config_data["num_sims"]==num_sims_scaling]

            campaign_config_data["seq_total"] = sequential_total

            speedup = campaign_config_data["seq_total"] / campaign_config_data["manager.general.environment_time"]
            efficiency = speedup / (num_nodes * num_cores)
            avg_efficiency = efficiency.mean()
            std_efficiency = efficiency.std()

            avg_efficiencies += [avg_efficiency]
            std_efficiencies += [std_efficiency]

        plt.errorbar(num_sims_list, avg_efficiencies, yerr=std_efficiencies, zorder=3, label="Nodes: {}".format(num_nodes))

    plt.title("Efficiciency for an increasing number of evaluations")
    plt.xlabel("Number of Evaluations")
    plt.ylabel("Efficiency")
    plt.ylim(ymin=0, ymax=1.15)
    plt.xticks(ticks=num_sims_list)
    plt.grid(zorder=-1.0)
    plt.legend()

    out_path_base = os.path.join(experiments_path, "figures" , "campaign_{}".format(model))
    os.makedirs(out_path_base, exist_ok=True)
    plt.savefig(os.path.join(out_path_base, "{}.png".format(savefig)))
    plt.savefig(os.path.join(out_path_base, "{}.pdf".format(savefig)))
    # plt.show()


def get_campaign_latex_table(experiments_path, model, num_nodes_list, num_workers_list, num_sims_list, savefig, caption, label):
    campaign_data = get_campaign_workflow_data(experiments_path, model)
    sims_runtime_data = get_campaign_sims_data(experiments_path, model)


    campaign_data = campaign_data.loc[campaign_data["num_nodes"].isin(num_nodes_list)]
    campaign_data = campaign_data.loc[campaign_data["num_workers"].isin(num_workers_list)]
    campaign_data = campaign_data.loc[campaign_data["num_sims"].isin(num_sims_list)]

    sims_runtime_data = sims_runtime_data.loc[sims_runtime_data["num_nodes"].isin(num_nodes_list)]
    sims_runtime_data = sims_runtime_data.loc[sims_runtime_data["num_workers"].isin(num_workers_list)]
    sims_runtime_data = sims_runtime_data.loc[sims_runtime_data["num_sims"].isin(num_sims_list)]

    campaign_data = campaign_data[["manager.general.environment_time", "num_nodes", "num_workers", "num_sims"]]
    campaign_data = campaign_data.rename(columns={
                        "manager.general.environment_time": "Environment",
                        "num_nodes": "Nodes",
                        "num_sims": "Evals",
                        "num_workers": "Workers"})

    sims_runtime_data = sims_runtime_data[["compilation.compile_time", "simulation.simulation_time", "general.dask_worker_time", "num_nodes", "num_workers", "num_sims"]]
    sims_runtime_data = sims_runtime_data.rename(columns={
                        "compilation.compile_time": "Compilation",
                        "simulation.simulation_time": "Simulation",
                        "general.dask_worker_time": "Worker",
                        "num_nodes": "Nodes",
                        "num_sims": "Evals",
                        "num_workers": "Workers"})

    if len(num_workers_list) == 1:
        index = ["Nodes", "Evals"]
    else:
        index = ["Nodes", "Workers", "Evals"]


    campaign_data = campaign_data.set_index(index)
    campaign_data[("Environment", "Mean")] = campaign_data.groupby(index)["Environment"].transform(np.mean)
    campaign_data[("Environment", "Std")] = campaign_data.groupby(index)["Environment"].transform(np.std)
    campaign_data = campaign_data[[("Environment", "Mean"), ("Environment", "Std")]]
    campaign_data = campaign_data.drop_duplicates()
    campaign_data.columns = pd.MultiIndex.from_tuples(campaign_data.columns, names=["Timeframe", "Measure (s)"])
    campaign_data = campaign_data.sort_index()

    sims_runtime_data = sims_runtime_data.set_index(index)
    time_frames = ["Simulation", "Compilation", "Worker"]

    for time_frame in time_frames:
        sims_runtime_data[(time_frame, "Mean")] = sims_runtime_data.groupby(index)[time_frame].transform(np.mean)
        sims_runtime_data[(time_frame, "Std")] = sims_runtime_data.groupby(index)[time_frame].transform(np.std)

    sims_runtime_data = sims_runtime_data[[(time_frame, measure) for time_frame in time_frames for measure in ["Mean", "Std"]]]
    sims_runtime_data = sims_runtime_data.drop_duplicates()
    sims_runtime_data.columns = pd.MultiIndex.from_tuples(sims_runtime_data.columns, names=["Timeframe", "Measure (s)"])
    sims_runtime_data = sims_runtime_data.sort_index()

    combined_data = campaign_data.join(sims_runtime_data)


    out_path_base = os.path.join(experiments_path, "tables" , "campaign_{}".format(model))
    os.makedirs(out_path_base, exist_ok=True)

    with open(os.path.join(out_path_base, "environment_{}.tex".format(savefig)), "w") as f:
        f.write(campaign_data.to_latex(float_format="{:.2f}".format, longtable=True, caption=caption, label=label))

    with open(os.path.join(out_path_base, "sims_{}.tex".format(savefig)), "w") as f:
        f.write(sims_runtime_data.to_latex(float_format="{:.2f}".format, longtable=True, caption=caption, label=label))

    with open(os.path.join(out_path_base, "combined_{}.tex".format(savefig)), "w") as f:
        f.write(combined_data.to_latex(float_format="{:.2f}".format, longtable=True, caption=caption, label=label))


def get_sequential_inet_latex_table(experiments_path, models, baseline_names, savefig, caption, label):
    combined_data = pd.DataFrame()

    for model, baseline_name in zip(models, baseline_names):
        sequential_data = get_sequential_data(experiments_path, model)

        sequential_data["model"] = baseline_name

        sequential_data = sequential_data[["compilation.compile_time", "simulation.simulation_time", "model"]]
        sequential_data = sequential_data.rename(columns={
                            "compilation.compile_time": "Compilation",
                            "simulation.simulation_time": "Simulation",
                            "model": "Baseline"})

        sequential_data = sequential_data.set_index("Baseline")

        time_frames = ["Simulation", "Compilation"]

        for time_frame in time_frames:
            sequential_data[(time_frame, "Mean")] = sequential_data.groupby("Baseline")[time_frame].transform(np.mean)
            sequential_data[(time_frame, "Std")] = sequential_data.groupby("Baseline")[time_frame].transform(np.std)

        sequential_data = sequential_data[[(time_frame, measure) for time_frame in time_frames for measure in ["Mean", "Std"]]]
        sequential_data = sequential_data.drop_duplicates()
        sequential_data.columns = pd.MultiIndex.from_tuples(sequential_data.columns, names=["Timeframe", "Measure (s)"])

        combined_data = pd.concat([combined_data, sequential_data])

    out_path_base = os.path.join(experiments_path, "tables" , "sequential")
    os.makedirs(out_path_base, exist_ok=True)

    with open(os.path.join(out_path_base, "combined_{}.tex".format(savefig)), "w") as f:
        f.write(combined_data.to_latex(float_format="{:.2f}".format, longtable=True, caption=caption, label=label))



def main():
    experiments_path = "/USER/experiments_das6_uva/"

# ---------------------------------------------------------------------------------------------------------------

    # model = "inet_lans"
    # num_nodes_list = [1, 2, 3, 4]
    # num_workers_list = [1]
    # num_sims_list = [32, 64, 96, 128, 160, 192, 224, 256]
    # savefig = "n1234_w1_sall"
    # caption = "Runtime data of simulation campaigns with the INET-LANS model. All configurations were set to a single worker process per compute node."
    # label = "tab:appendix:inet_single_worker"

    # get_campaign_latex_table(experiments_path, model, num_nodes_list, num_workers_list, num_sims_list, savefig, caption, label)


    # model = "inet_lans"
    # num_nodes_list = [1, 2, 3, 4]
    # num_workers_list = [1, 2, 4, 8, 16, 32]
    # num_sims_list = [32, 64, 96, 128]
    # savefig = "n1234_wall_s3264"
    # caption = "Runtime data of simulation campaigns with the INET-LANS model. All non-single worker runtime data and the corresponding single worker runtime data."
    # label = "tab:appendix:inet_multi_worker"

    # get_campaign_latex_table(experiments_path, model, num_nodes_list, num_workers_list, num_sims_list, savefig, caption, label)


    # model = "hyper_inet_lans"
    # num_nodes_list = [1, 2, 3, 4]
    # num_workers_list = [1]
    # num_sims_list = [32, 64, 96, 128, 160, 192, 224, 256]
    # savefig = "n1234_w1_sall_hyper"
    # caption = "Runtime data of SMT-enabled simulation campaigns with the INET-LANS model. All configurations were set to a single worker process per compute node."
    # label = "tab:appendix:hyper_inet_single_worker"

    # get_campaign_latex_table(experiments_path, model, num_nodes_list, num_workers_list, num_sims_list, savefig, caption, label)


# ---------------------------------------------------------------------------------------------------------------


    # model = "pdes_comm"
    # num_nodes_list = [1, 2, 3, 4]
    # num_workers_list = [1]
    # num_sims_list = [4, 8, 12, 16, 20, 24, 28, 32]
    # savefig = "pdes_comm_n1234_w1_sall"
    # caption = "Runtime data of simulation campaigns with the PDES-enabled communication intensive CQN model. All configurations were set to a single worker process per compute node."
    # label = "tab:appendix:pdes_comm_single_worker"

    # get_campaign_latex_table(experiments_path, model, num_nodes_list, num_workers_list, num_sims_list, savefig, caption, label)


    # model = "pdes_comp"
    # num_nodes_list = [1, 2, 3, 4]
    # num_workers_list = [1]
    # num_sims_list = [4, 8, 12, 16, 20, 24, 28, 32]
    # savefig = "pdes_comp_n1234_w1_sall"
    # caption = "Runtime data of simulation campaigns with the PDES-enabled computation intensive CQN model. All configurations were set to a single worker process per compute node."
    # label = "tab:appendix:pdes_comp_single_worker"

    # get_campaign_latex_table(experiments_path, model, num_nodes_list, num_workers_list, num_sims_list, savefig, caption, label)


    # model = "pdes_sub"
    # num_nodes_list = [1, 2, 3, 4]
    # num_workers_list = [1]
    # num_sims_list = [4, 8, 12, 16, 20, 24, 28, 32]
    # savefig = "pdes_sub_n1234_w1_sall"
    # caption = "Runtime data of simulation campaigns with the PDES-enabled subsystem intensive CQN model. All configurations were set to a single worker process per compute node."
    # label = "tab:appendix:pdes_sub_single_worker"

    # get_campaign_latex_table(experiments_path, model, num_nodes_list, num_workers_list, num_sims_list, savefig, caption, label)


# ---------------------------------------------------------------------------------------------------------------


    # model = "seq_comm"
    # num_nodes_list = [1, 2, 3, 4]
    # num_workers_list = [1]
    # num_sims_list = [4, 8, 12, 16, 20, 24, 28, 32, 64, 96, 128, 256]
    # savefig = "seq_comm_n1234_w1_sall"
    # caption = "Runtime data of simulation campaigns with the sequential communication intensive CQN model. All configurations were set to a single worker process per compute node."
    # label = "tab:appendix:seq_comm_single_worker"

    # get_campaign_latex_table(experiments_path, model, num_nodes_list, num_workers_list, num_sims_list, savefig, caption, label)


    # model = "seq_comp"
    # num_nodes_list = [1, 2, 3, 4]
    # num_workers_list = [1]
    # num_sims_list = [4, 8, 12, 16, 20, 24, 28, 32, 64, 96, 128, 256]
    # savefig = "seq_comp_n1234_w1_sall"
    # caption = "Runtime data of simulation campaigns with the sequential computation intensive CQN model. All configurations were set to a single worker process per compute node."
    # label = "tab:appendix:seq_comp_single_worker"

    # get_campaign_latex_table(experiments_path, model, num_nodes_list, num_workers_list, num_sims_list, savefig, caption, label)


    # model = "seq_sub"
    # num_nodes_list = [1, 2, 3, 4]
    # num_workers_list = [1]
    # num_sims_list = [4, 8, 12, 16, 20, 24, 28, 32, 64, 96, 128, 256]
    # savefig = "seq_sub_n1234_w1_sall"
    # caption = "Runtime data of simulation campaigns with the seqeuntial subsystem intensive CQN model. All configurations were set to a single worker process per compute node."
    # label = "tab:appendix:seq_sub_single_worker"

    # get_campaign_latex_table(experiments_path, model, num_nodes_list, num_workers_list, num_sims_list, savefig, caption, label)



# ---------------------------------------------------------------------------------------------------------------


    # models = ["inet_lans", "multi_inet_lans"]
    # baseline_names = ["Sequential", "Sequential Multi"]
    # savefig = "seq_inet"
    # caption = "Runtime data of sequential baselines with the INET-LANS model."
    # label = "tab:appendix:seq_baseline_inet"

    # get_sequential_inet_latex_table(experiments_path, models, baseline_names, savefig, caption, label)


    # models = ["seq_comm", "multi_seq_comm", "pdes_comm", "multi_pdes_comm"]
    # baseline_names = ["Sequential", "Sequential Multi", "Sequential PDES", "Sequential PDES Multi"]
    # savefig = "seq_comm"
    # caption = "Runtime data of baselines with the communication intensive CQN model."
    # label = "tab:appendix:seq_baseline_comm"

    # get_sequential_inet_latex_table(experiments_path, models, baseline_names, savefig, caption, label)

    # models = ["seq_comp", "multi_seq_comp", "pdes_comp", "multi_pdes_comp"]
    # baseline_names = ["Sequential", "Sequential Multi", "Sequential PDES", "Sequential PDES Multi"]
    # savefig = "seq_comp"
    # caption = "Runtime data of baselines with the computation intensive CQN model."
    # label = "tab:appendix:seq_baseline_comp"

    # get_sequential_inet_latex_table(experiments_path, models, baseline_names, savefig, caption, label)

    # models = ["seq_sub", "multi_seq_sub", "pdes_sub", "multi_pdes_sub"]
    # baseline_names = ["Sequential", "Sequential Multi", "Sequential PDES", "Sequential PDES Multi"]
    # savefig = "seq_sub"
    # caption = "Runtime data of baselines with the subsystem intensive CQN model."
    # label = "tab:appendix:seq_baseline_sub"

    # get_sequential_inet_latex_table(experiments_path, models, baseline_names, savefig, caption, label)




# ---------------------------------------------------------------------------------------------------------------


    # model = "inet_lans"
    # num_nodes_list = [1, 2, 3, 4]
    # num_workers = 1
    # num_sims = 32
    # num_cores = 32
    # scaling_list = ["weak", "strong"]

    # for scaling in scaling_list:
    #     savefig = "s{}_{}".format(num_sims, scaling)
    #     campaign_visualise_speedup_efficiency(experiments_path, savefig, model, scaling, num_nodes_list, num_workers, num_sims, num_cores)


# ---------------------------------------------------------------------------------------------------------------


    # model = "inet_lans"
    # num_nodes_list = [1, 2, 3, 4]
    # num_workers = 1
    # num_sims = 64
    # num_cores = 32
    # scaling_list = ["weak", "strong"]

    # for scaling in scaling_list:
    #     savefig = "s{}_{}_hyper".format(num_sims, scaling)
    #     campaign_visualise_speedup_efficiency(experiments_path, savefig, model, scaling, num_nodes_list, num_workers, num_sims, num_cores, hyper=True)


# ---------------------------------------------------------------------------------------------------------------

    # model = "inet_lans"
    # num_nodes_list = [1, 2, 3, 4]
    # num_workers = 1
    # num_sims_list = [32, 64, 96, 128, 160, 192, 224, 256]
    # num_cores = 32
    # scaling_list = ["strong"]
    # baselines = ["sequential", "sequential_multi"]

    # for baseline in baselines:
    #     for scaling in scaling_list:
    #         savefig = "efficiency_sims_{}_base_{}".format(scaling, baseline)
    #         visualise_sims_efficiency(experiments_path, savefig, model, scaling, num_nodes_list, num_workers, num_sims_list, num_cores, baseline=baseline)
    #         savefig = "speedup_sims_{}_base_{}".format(scaling, baseline)
    #         visualise_sims_speedup(experiments_path, savefig, model, scaling, num_nodes_list, num_workers, num_sims_list, num_cores, baseline=baseline)


# ---------------------------------------------------------------------------------------------------------------

    # model = "inet_lans"
    # num_nodes_list = [1, 2, 3, 4]
    # num_workers = 1
    # num_sims_list = [32, 64, 96, 128, 160, 192, 224, 256]
    # num_cores = 32
    # scaling_list = ["strong"]
    # baselines = ["sequential", "sequential_multi"]

    # for baseline in baselines:
    #     for scaling in scaling_list:
    #         savefig = "efficiency_sims_{}_base_{}_hyper".format(scaling, baseline)
    #         visualise_sims_efficiency(experiments_path, savefig, model, scaling, num_nodes_list, num_workers, num_sims_list, num_cores, baseline=baseline, hyper=True)
    #         savefig = "speedup_sims_{}_base_{}_hyper".format(scaling, baseline)
    #         visualise_sims_speedup(experiments_path, savefig, model, scaling, num_nodes_list, num_workers, num_sims_list, num_cores, baseline=baseline, hyper=True)


# ---------------------------------------------------------------------------------------------------------------


    # baselines = ["seq", "seq_multi"]
    # labels = ["Sequential", "Sequential multi"]
    # models = ["inet_lans", "inet_lans"]
    # hypers = [True, False]
    # num_nodes_list = [1, 2, 3, 4]
    # num_workers = 1
    # num_sims = 256
    # num_cores = 32
    # scaling_list = ["weak", "strong"]

    # for scaling in scaling_list:
    #     savefig = "s{}_{}".format(num_sims, scaling)
    #     campaign_visualise_relative_speedup(experiments_path, savefig, scaling, baselines, labels, models, hypers, num_nodes_list, num_workers, num_sims, num_cores)

# ---------------------------------------------------------------------------------------------------------------


    # baselines = ["seq", "seq_multi"]
    # labels = ["Sequential", "Sequential multi"]
    # models = ["inet_lans", "inet_lans"]
    # hypers = [True, False]
    # num_nodes_list = [1, 2, 3, 4]
    # num_workers = 1
    # num_sims_list = [32, 64, 96, 128, 160, 192, 224, 256]
    # num_cores = 32
    # scaling_list = ["strong"]

    # for scaling in scaling_list:
    #     savefig = "{}".format(scaling)
    #     campaign_visualise_relative_speedup_line(experiments_path, savefig, scaling, baselines, labels, models, hypers, num_nodes_list, num_workers, num_sims_list, num_cores)



# ---------------------------------------------------------------------------------------------------------------

    # model = "inet_lans"
    # num_nodes_list = [2]
    # num_workers_list = [1]
    # num_sims_list = [64, 96, 128]

    # for num_nodes in num_nodes_list:
    #     for num_workers in num_workers_list:
    #         for num_sims in num_sims_list:
    #             savefig = "gantt_n{}_w{}_s{}".format(num_nodes, num_workers, num_sims)
    #             visualise_gantt(experiments_path, savefig, model, num_nodes, num_workers, num_sims)


    # model = "pdes_comm"
    # num_nodes_list = [4]
    # num_workers_list = [1]
    # num_sims_list = [16]

    # for num_nodes in num_nodes_list:
    #     for num_workers in num_workers_list:
    #         for num_sims in num_sims_list:
    #             savefig = "gantt_n{}_w{}_s{}".format(num_nodes, num_workers, num_sims)
    #             visualise_gantt(experiments_path, savefig, model, num_nodes, num_workers, num_sims)


# ---------------------------------------------------------------------------------------------------------------


    # model = "inet_lans"
    # num_nodes_list = [1, 2, 3, 4]
    # num_workers_list = [1, 2, 4, 8, 16, 32]
    # num_sims_list = [32]

    # # savefig = "worker_size_n1234_s32"
    # # scaling = "weak"
    # # visualise_worker_size(experiments_path, savefig, model, num_nodes_list, num_workers_list, num_sims_list)

    # scaling_list = ["weak", "strong"]

    # for scaling in scaling_list:
    #     for num_sims in num_sims_list:
    #         savefig = "worker_size_n1234_s32_{}".format(scaling)
    #         visualise_worker_size(experiments_path, savefig, model, scaling, num_nodes_list, num_workers_list, [num_sims])


    # # for num_nodes in num_nodes_list:
    # #     savefig = "worker_size_n{}_s32_64".format(num_nodes)
    # #     visualise_worker_size(experiments_path, savefig, model, [num_nodes], num_workers_list, num_sims_list)

# ---------------------------------------------------------------------------------------------------------------



    # base_models = ["sub", "comm", "comp"]
    # num_nodes_list = [1, 2, 3, 4]
    # num_workers = 1
    # num_sims = 4
    # scaling_list = ["strong", "weak"]
    # num_cores = 32

    # for base_model in base_models:
    #     for scaling in scaling_list:
    #         print(base_model, scaling)
    #         savefig = "pdes_{}_s{}_{}".format(base_model, num_sims, scaling)
    #         pdes_visualise_speedup_efficiency(experiments_path, savefig, base_model, scaling, num_nodes_list, num_workers, num_sims, num_cores)

# ---------------------------------------------------------------------------------------------------------------


    # num_nodes_list = [1, 2, 3, 4]
    # num_workers = 1
    # num_sims = 4
    # scaling_list = ["strong", "weak"]
    # num_cores = 32

    # for scaling in scaling_list:
    #     savefig = "pdes_s{}_{}".format(num_sims, scaling)
    #     pdes_visualise_relative_speedup(experiments_path, savefig, scaling, num_nodes_list, num_workers, num_sims, num_cores)


# ---------------------------------------------------------------------------------------------------------------



    # num_nodes_list = [1, 2, 3, 4]
    # num_workers = 1
    # num_sims_list = [4, 8, 12, 16, 20, 24, 28, 32]
    # scaling_list = ["strong"]
    # num_cores = 32

    # for scaling in scaling_list:
    #     savefig = "pdes_{}".format(scaling)
    #     pdes_visualise_relative_speedup_lines(experiments_path, savefig, scaling, num_nodes_list, num_workers, num_sims_list, num_cores)


# ---------------------------------------------------------------------------------------------------------------



    # model_list = ["pdes_comm", "pdes_comp", "pdes_sub"]
    # num_nodes_list = [1, 2, 3, 4]
    # num_workers = 1
    # num_sims_list = [4, 8, 12, 16, 20, 24, 28, 32]
    # num_cores = 32
    # scaling_list = ["strong"]
    # baselines = ["sequential", "sequential_multi"]

    # for model in model_list:
    #     for baseline in baselines:
    #         for scaling in scaling_list:
    #             savefig = "efficiency_sims_{}_base_{}".format(scaling, baseline)
    #             visualise_sims_efficiency(experiments_path, savefig, model, scaling, num_nodes_list, num_workers, num_sims_list, num_cores, baseline=baseline)
    #             savefig = "speedup_sims_{}_base_{}".format(scaling, baseline)
    #             visualise_sims_speedup(experiments_path, savefig, model, scaling, num_nodes_list, num_workers, num_sims_list, num_cores, baseline=baseline)



# ---------------------------------------------------------------------------------------------------------------


    # models = ["seq_comm", "seq_comp", "seq_sub"]
    # num_nodes_list = [1, 2, 3, 4]
    # num_workers = 1
    # num_sims = 32
    # num_cores = 32
    # scaling_list = ["weak", "strong"]

    # for model in models:
    #     for scaling in scaling_list:
    #         savefig = "{}_s{}_{}".format(model, num_sims, scaling)
    #         campaign_visualise_speedup_efficiency(experiments_path, savefig, model, scaling, num_nodes_list, num_workers, num_sims, num_cores)




# ---------------------------------------------------------------------------------------------------------------

    results_path = os.path.join(experiments_path, "results")
    experiment_folders = [os.path.join(results_path, exp) for exp in sorted(os.listdir(results_path)) if os.path.isdir(os.path.join(results_path, exp))]
    total_time = 0
    for exp_path in experiment_folders:
        sim_runtime_path = os.path.join(exp_path, "sims_runtime.csv")
        sim_runtime_data = pd.read_csv(sim_runtime_path)
        sim_time = sim_runtime_data["simulation.simulation_time"] + sim_runtime_data["compilation.compile_time"]
        total_time += sim_time.sum()

    print(timedelta(seconds=total_time))

if __name__ == '__main__':
    main()