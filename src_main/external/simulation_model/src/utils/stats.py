import time
import os
import json

from src_main.tools.local_parallelization.rpc import requires_main_process, callable_from_main

class Stats:
    stats = None

    @requires_main_process
    def __init__(self, pre_stats=None, file_path=None):
        if pre_stats and isinstance(pre_stats, dict):
            self.stats = pre_stats
        else:
            self.stats = {}
        self.file_path = file_path

    @requires_main_process
    def set_file_path(self, file_path):
        self.file_path = file_path

    @staticmethod
    def merge_dicts(dict1, dict2, path=None):
        # based on: https://stackoverflow.com/a/7205107
        if path is None: path = []
        for key in dict2:
            if key in dict1:
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    Stats.merge_dicts(dict1[key], dict2[key], path + [str(key)])
                elif dict1[key] == dict2[key]:
                    pass # same leaf value
                else:
                    raise Exception("Conflict at {}".format(".".join(path + [str(key)])))
            else:
                dict1[key] = dict2[key]
        return dict1

    @staticmethod
    def update_dict(current_dict, update_dict):
        test = {}
        for k, v in update_dict.items():
            if isinstance(v, dict):
                current_dict[k] = Stats.update_dict(current_dict.get(k, {}), v)
            else:
                current_dict[k] = v

        return current_dict

    @staticmethod
    def create_dict_from_args_val(val, *args):
        cur_dict = {}
        for i, arg in enumerate(reversed(args)):
            if i == 0:
                cur_dict = {arg: val}
            else:
                cur_dict = {arg: cur_dict}

        return cur_dict

    @requires_main_process
    def has_stat(self, *args):
        cur = self.stats
        for arg in args:
            if arg in cur:
                cur = cur[arg]
            else:
                return False

        if isinstance(cur, dict):
            return False
        else:
            return True

    @requires_main_process
    def get_stat(self, *args):
        if self.has_stat(*args):
            cur = self.stats
            for arg in args:
                cur = cur[arg]
            return cur
        else:
            return None

    @requires_main_process
    def record_stat(self, val, *args):
        merge_dict = Stats.create_dict_from_args_val(val, *args)
        self.stats = Stats.update_dict(self.stats, merge_dict)

    @requires_main_process
    def record_time_stat(self, *args):
        record_time = time.time()
        self.record_stat(record_time, *args)
        return record_time

    @requires_main_process
    def inc_stat(self, *args):
        self.add_stat(1, *args)

    @requires_main_process
    def add_stat(self, val, *args):
        old_val = self.get_stat(*args)
        new_val = val

        if old_val:
            new_val = val + old_val

        self.record_stat(new_val, *args)

    @requires_main_process
    def get_stats_dict(self):
        return self.stats

    @requires_main_process
    def write_stats_to_file(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        elif self.file_path is None:
            self.file_path = file_path
        else:
            raise ValueError("No file path set for writing statistic file.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        stats_file = self.stats
        stats_file["time_of_save"] = time.time()
        with open(file_path, "w",  encoding='utf-8') as json_file:
            json.dump(self.stats, json_file)